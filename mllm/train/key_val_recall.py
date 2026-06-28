"""Key-value recall dataset (structured-extraction family §2.1).

Automated, but grounded in *real* Wikipedia text: each record is a list of
``key: value`` pairs whose **keys** and **values** are real words / spans sampled
from a Wikipedia article. The decoder is prompted with one key and must emit the
paired value. The value lives *only* in the encoded record (priors cannot supply
it), so the cheapest path to low loss runs through the context embeddings.

Testability split
-----------------
All randomness lives in :meth:`KeyValRecallTokenizer.sample_spec` (producing a
:class:`KeyValSpec`). :meth:`KeyValRecallTokenizer.realize` is a pure,
deterministic function of ``(spec, WordFeed)`` — words are consumed
*sequentially* from the feed. ``build(text)`` is the thin wrapper that samples a
spec and realizes it.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

from mllm.data.utils import TokensSubsetV2
from mllm.train.extraction_common import (
    BaseRecallDataset, WordFeed, assemble_subset, create_recall_dataloader,
    dec_end_token_id, enc_to_dec, make_pool, sanitize_text, word_feed_from_text,
)


@dataclass(kw_only=True)
class KeyValRecallCfg:
    # Number of key:value pairs per record (the capacity knob).
    min_pairs: int = 4
    max_pairs: int = 12
    # Value length in *words* (a value is 1..value_max_words consecutive real words).
    value_min_words: int = 1
    value_max_words: int = 3
    # Key length in *words*. Multi-word keys stay distinctive so a frequent
    # stopword (``is`` / ``in`` / ``the``) cannot collide across many pairs.
    key_min_words: int = 3
    key_max_words: int = 4
    # Minimum decoded length (chars) for a token-run to qualify as a key word.
    min_key_chars: int = 2
    # Probability of emitting a composite (whole-record dump) target.
    composite_prob: float = 0.0
    # When True, pack each record up to the token budget by emitting max_pairs
    # pairs (realize stops once the budget is hit). fill_frac is unused here (the
    # greedy pack already fills the chunk) but kept for cfg parity.
    fill_to_budget: bool = False
    fill_frac: float = 0.85


@dataclass(kw_only=True)
class KeyValSpec:
    """Fully describes one key-value sample; consumed deterministically."""
    n_pairs: int
    kv_sep_idx: int
    pair_sep_idx: int
    value_word_counts: List[int]
    key_word_counts: List[int]
    query_pair_idx: int
    # None -> single-value target; 'record' -> whole-record composite target.
    composite: Optional[str] = None


class KeyValRecallTokenizer:
    """Builds one ``TokensSubsetV2`` key-value-recall sample per input text."""

    def __init__(
            self, tkz_enc: PreTrainedTokenizer, max_len: int, cfg: Optional[KeyValRecallCfg] = None,
            n_special_toks: int = 1000, tkz_dec: Optional[PreTrainedTokenizer] = None,
            seed: Optional[int] = None,
    ):
        self.tkz_enc = tkz_enc
        self.tkz_dec = tkz_dec if tkz_dec is not None else tkz_enc
        self.same_vocab = self.tkz_enc is self.tkz_dec or len(self.tkz_enc) == len(self.tkz_dec)
        self.max_len = max_len
        self.cfg = cfg if cfg is not None else KeyValRecallCfg()
        self.rng = np.random.default_rng(seed)
        self.pool = make_pool(self.rng, n_special_toks, len(self.tkz_enc))

        # Separator token sequences (encoder vocab). Index chosen in the spec so the
        # model cannot learn a fixed-format position (anti-shortcut, plan §6).
        self.kv_sep_choices = [
            self.tkz_enc(s, add_special_tokens=False).input_ids for s in (':', ' :', ' =', ' -')
        ]
        self.pair_sep_choices = [
            self.tkz_enc(s, add_special_tokens=False).input_ids for s in (';', ' ;', ' .', ' |')
        ]
        # Quote token(s) (encoder vocab) wrapping every key and value so their
        # boundaries are unambiguous: ``"key" : "value"``.
        self.quote = self.tkz_enc('"', add_special_tokens=False).input_ids

        # Decoder-vocab prompt template: 'Key: "<key>". Retrieve its value.'
        td = self.tkz_dec
        self.prompt_beg = td('Key: "', add_special_tokens=False).input_ids
        self.prompt_end = td('". Retrieve its value.', add_special_tokens=False).input_ids
        self.prompt_record = td('Dump the whole record as key: value pairs.', add_special_tokens=False).input_ids
        self.dec_end_token_id = dec_end_token_id(td)

    # --- helpers -------------------------------------------------------------
    def _is_key_word(self, txt: str) -> bool:
        return len(txt) >= self.cfg.min_key_chars and any(c.isalnum() for c in txt)

    # --- random: spec sampling ----------------------------------------------
    def sample_spec(self) -> KeyValSpec:
        cfg = self.cfg
        rng = self.rng
        n_pairs = cfg.max_pairs if cfg.fill_to_budget else int(rng.integers(cfg.min_pairs, cfg.max_pairs + 1))
        composite = None
        if float(rng.random()) < cfg.composite_prob:
            composite = 'record'
        return KeyValSpec(
            n_pairs=n_pairs,
            kv_sep_idx=int(rng.integers(len(self.kv_sep_choices))),
            pair_sep_idx=int(rng.integers(len(self.pair_sep_choices))),
            value_word_counts=[
                int(rng.integers(cfg.value_min_words, cfg.value_max_words + 1))
                for _ in range(n_pairs)
            ],
            key_word_counts=[
                int(rng.integers(cfg.key_min_words, cfg.key_max_words + 1))
                for _ in range(n_pairs)
            ],
            query_pair_idx=int(rng.integers(n_pairs)),
            composite=composite,
        )

    # --- deterministic: realization -----------------------------------------
    def realize(self, spec: KeyValSpec, feed: WordFeed, ids_src: Optional[List[int]] = None) -> TokensSubsetV2:
        cfg = self.cfg
        budget = self.max_len - 2  # reserve CLS + SEP
        kv_sep = self.kv_sep_choices[spec.kv_sep_idx]
        pair_sep = self.pair_sep_choices[spec.pair_sep_idx]
        q = self.quote
        quote_len = 4 * len(q)  # two quotes around the key + two around the value

        pairs: List[tuple] = []  # (key_ids, key_text, value_ids)
        used_len = 0
        used_keys: set = set()
        for p in range(spec.n_pairs):
            kc = spec.key_word_counts[p] if p < len(spec.key_word_counts) else cfg.key_min_words
            _, key_text = feed.next_distinct_phrase(kc, used_keys, is_valid=self._is_key_word)
            key_text = sanitize_text(key_text)
            key_ids = self.tkz_enc(key_text, add_special_tokens=False).input_ids

            c = spec.value_word_counts[p] if p < len(spec.value_word_counts) else cfg.value_min_words
            _, value_text = feed.next_span(c)
            value_text = sanitize_text(value_text)
            value_ids = self.tkz_enc(value_text, add_special_tokens=False).input_ids
            if not key_ids or not value_ids:
                continue

            add_len = quote_len + len(key_ids) + len(kv_sep) + len(value_ids) + (len(pair_sep) if pairs else 0)
            if pairs and used_len + add_len > budget:
                break
            if not pairs and quote_len + len(key_ids) + len(kv_sep) + len(value_ids) > budget:
                keep = max(1, budget - quote_len - len(key_ids) - len(kv_sep))
                value_ids = value_ids[:keep]
                add_len = quote_len + len(key_ids) + len(kv_sep) + len(value_ids)
            pairs.append((key_ids, key_text, value_ids))
            used_len += add_len

        # Serialize record (encoder vocab): "k1" sep "v1" PAIRSEP "k2" sep "v2" ...
        record: List[int] = []
        for i, (k, _kt, v) in enumerate(pairs):
            if i > 0:
                record.extend(pair_sep)
            record.extend(q)
            record.extend(k)
            record.extend(q)
            record.extend(kv_sep)
            record.extend(q)
            record.extend(v)
            record.extend(q)
        record = record[:budget]

        # Queried pair (clamped: budget may have dropped trailing pairs).
        qi = spec.query_pair_idx % len(pairs)
        _key_ids_q, key_text_q, value_ids_q = pairs[qi]
        key_dec = self.tkz_dec(key_text_q, add_special_tokens=False).input_ids

        if spec.composite == 'record':
            prompt_toks = list(self.prompt_record)
            target_enc = list(record)
            target_dec = enc_to_dec(record, self.tkz_enc, self.tkz_dec, self.same_vocab)
        else:
            prompt_toks = self.prompt_beg + key_dec + self.prompt_end
            target_enc = list(value_ids_q)
            target_dec = enc_to_dec(value_ids_q, self.tkz_enc, self.tkz_dec, self.same_vocab)

        prompt = self.tkz_dec.decode(prompt_toks, skip_special_tokens=False)

        return assemble_subset(
            ids_src=ids_src if ids_src is not None else list(record),
            rec_ids=record,
            prompt=prompt,
            prompt_toks=prompt_toks,
            tkz_enc=self.tkz_enc,
            tkz_dec=self.tkz_dec,
            same_vocab=self.same_vocab,
            end_token_id=self.dec_end_token_id,
            target_enc=target_enc,
            target_dec=target_dec,
        )

    # --- wrapper -------------------------------------------------------------
    def build(self, text: str) -> TokensSubsetV2:
        feed, ids_src = word_feed_from_text(text, self.tkz_enc, self.pool, rng=self.rng)
        spec = self.sample_spec()
        return self.realize(spec, feed, ids_src=ids_src)

    def __call__(self, texts: List[str]) -> List[TokensSubsetV2]:
        return [self.build(t) for t in texts]


class KeyValRecallDataset(BaseRecallDataset):
    """Wraps a HuggingFace ``text`` dataset (e.g. Wikipedia) and yields
    ``MaskedCiteBatch`` objects with key-value-recall content."""

    def __init__(
            self, dataset: Dataset, tkz_enc: PreTrainedTokenizer, max_seq_len: int,
            n_special_toks: int = 1000, cfg: Optional[KeyValRecallCfg] = None,
            device: Optional[torch.device] = None, tkz_dec: Optional[PreTrainedTokenizer] = None,
            seed: Optional[int] = None,
    ):
        builder = KeyValRecallTokenizer(
            tkz_enc, max_len=max_seq_len, cfg=cfg, n_special_toks=n_special_toks,
            tkz_dec=tkz_dec if tkz_dec is not None else tkz_enc, seed=seed,
        )
        super().__init__(dataset, tkz_enc, max_seq_len, builder, device=device, tkz_dec=tkz_dec)


def create_key_val_recall_dataloader(dataset: KeyValRecallDataset, batch_size: int):
    return create_recall_dataloader(dataset, batch_size, name='KeyValRecall')
