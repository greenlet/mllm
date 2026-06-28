"""JSON field recall dataset (structured-extraction family §2.2).

Automated, but grounded in *real* Wikipedia text: each sample builds a randomized
JSON object and asks for one field via JSONPath / dotted key / NL phrasing. The
answer is the selected value and must be read from the encoded JSON record.

Testability split
-----------------
``sample_spec`` decides the JSON *shape* (which fields are nested / arrays /
scalars, literals, separators, which path is queried) — all randomness lives
here, and it is content-agnostic. :meth:`JsonFieldRecallTokenizer.realize` is a
deterministic function of ``(spec, WordFeed)``: keys and string values are pulled
*sequentially* from the article's words. ``build`` resamples specs until the
serialized record fits the token budget.
"""
from dataclasses import dataclass
import json
from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

from mllm.data.utils import TokensSubsetV2
from mllm.train.extraction_common import (
    BaseRecallDataset, JsonNodeSpec, WordFeed, assemble_subset, build_json_value,
    build_to_budget, create_recall_dataloader, dec_end_token_id, json_path_strings,
    json_seps, json_target_text, make_pool, sample_json_shape, word_feed_from_text,
)


@dataclass(kw_only=True)
class JsonFieldRecallCfg:
    min_fields: int = 4
    max_fields: int = 10
    max_depth: int = 3
    max_array_len: int = 4
    value_min_words: int = 1
    value_max_words: int = 3
    nested_prob: float = 0.35
    array_prob: float = 0.20
    number_prob: float = 0.20
    bool_null_prob: float = 0.10
    min_key_chars: int = 2
    # Key length in *words*. Multi-word keys stay distinctive so a frequent
    # stopword cannot collide across fields / JSON paths.
    key_min_words: int = 3
    key_max_words: int = 4
    # Probability that the queried target is a composite subtree (object/array)
    # rather than a scalar leaf. Serialized with the record's separators so it
    # stays a verbatim substring.
    composite_prob: float = 0.0
    # When True, grow records to ~fill the chunk: keep the largest sampled record
    # that fits the token budget (early-stop at fill_frac * budget) instead of
    # returning the first that fits.
    fill_to_budget: bool = False
    fill_frac: float = 0.85


@dataclass(kw_only=True)
class JsonFieldSpec:
    root: JsonNodeSpec
    compact: bool
    key_words: int       # words per object key
    query_leaf_idx: int  # modular index into the realized leaf/composite list
    q_kind: int          # prompt phrasing: 0 jsonpath / 1 dotted / 2 NL
    composite: bool = False


class JsonFieldRecallTokenizer:
    def __init__(
            self, tkz_enc: PreTrainedTokenizer, max_len: int, cfg: Optional[JsonFieldRecallCfg] = None,
            n_special_toks: int = 1000, tkz_dec: Optional[PreTrainedTokenizer] = None,
            seed: Optional[int] = None,
    ):
        self.tkz_enc = tkz_enc
        self.tkz_dec = tkz_dec if tkz_dec is not None else tkz_enc
        self.same_vocab = self.tkz_enc is self.tkz_dec or len(self.tkz_enc) == len(self.tkz_dec)
        self.max_len = max_len
        self.cfg = cfg if cfg is not None else JsonFieldRecallCfg()
        self.rng = np.random.default_rng(seed)
        self.pool = make_pool(self.rng, n_special_toks, len(self.tkz_enc))
        self.dec_end_token_id = dec_end_token_id(self.tkz_dec)

    # --- random: spec sampling ----------------------------------------------
    def sample_spec(self) -> JsonFieldSpec:
        cfg = self.cfg
        rng = self.rng
        return JsonFieldSpec(
            root=sample_json_shape(rng, cfg),
            compact=bool(rng.integers(2)),
            key_words=int(rng.integers(cfg.key_min_words, cfg.key_max_words + 1)),
            query_leaf_idx=int(rng.integers(1 << 30)),
            q_kind=int(rng.integers(3)),
            composite=float(rng.random()) < cfg.composite_prob,
        )

    # --- deterministic: realization -----------------------------------------
    def realize(self, spec: JsonFieldSpec, feed: WordFeed, ids_src: Optional[List[int]] = None) -> Tuple[TokensSubsetV2, int]:
        budget = self.max_len - 2
        obj, leaves, _arrays, composites = build_json_value(spec.root, feed, self.cfg.min_key_chars, spec.key_words)

        seps = json_seps(spec.compact)
        rec_text = json.dumps(obj, ensure_ascii=True, separators=seps)
        rec_ids = self.tkz_enc(rec_text, add_special_tokens=False).input_ids
        rec_len = len(rec_ids)
        if rec_len > budget:
            rec_ids = rec_ids[:budget]
            rec_text = self.tkz_enc.decode(rec_ids, skip_special_tokens=True)

        candidates = composites if (spec.composite and composites) else leaves
        if not candidates:
            candidates = leaves or [([], obj)]
        n = len(candidates)
        path, value = candidates[spec.query_leaf_idx % n]
        tgt_text = json_target_text(value, spec.compact)
        for j in range(n):
            if tgt_text and tgt_text in rec_text:
                break
            path, value = candidates[(spec.query_leaf_idx + j + 1) % n]
            tgt_text = json_target_text(value, spec.compact)

        jp, dotted = json_path_strings(path)
        if spec.q_kind == 0:
            prompt = f'JSONPath: {jp}. Return value only.'
        elif spec.q_kind == 1:
            prompt = f'Key path: {dotted}. Return value only.'
        else:
            prompt = f'What is the value at path {dotted}? Return value only.'
        prompt_toks = self.tkz_dec(prompt, add_special_tokens=False).input_ids

        subset = assemble_subset(
            ids_src=ids_src if ids_src is not None else list(rec_ids),
            rec_ids=rec_ids,
            prompt=prompt,
            prompt_toks=prompt_toks,
            tkz_enc=self.tkz_enc,
            tkz_dec=self.tkz_dec,
            same_vocab=self.same_vocab,
            end_token_id=self.dec_end_token_id,
            target_text=tgt_text,
        )
        return subset, rec_len

    # --- wrapper -------------------------------------------------------------
    def build(self, text: str) -> TokensSubsetV2:
        feed, ids_src = word_feed_from_text(text, self.tkz_enc, self.pool, rng=self.rng)
        return build_to_budget(
            sample_spec=self.sample_spec,
            realize=self.realize,
            feed=feed,
            ids_src=ids_src,
            budget=self.max_len - 2,
            n_attempts=16,
            fill_to_budget=self.cfg.fill_to_budget,
            fill_frac=self.cfg.fill_frac,
        )

    def __call__(self, texts: List[str]) -> List[TokensSubsetV2]:
        return [self.build(t) for t in texts]


class JsonFieldRecallDataset(BaseRecallDataset):
    def __init__(
            self, dataset: Dataset, tkz_enc: PreTrainedTokenizer, max_seq_len: int,
            n_special_toks: int = 1000, cfg: Optional[JsonFieldRecallCfg] = None,
            device: Optional[torch.device] = None, tkz_dec: Optional[PreTrainedTokenizer] = None,
            seed: Optional[int] = None,
    ):
        builder = JsonFieldRecallTokenizer(
            tkz_enc, max_len=max_seq_len, cfg=cfg, n_special_toks=n_special_toks,
            tkz_dec=tkz_dec if tkz_dec is not None else tkz_enc, seed=seed,
        )
        super().__init__(dataset, tkz_enc, max_seq_len, builder, device=device, tkz_dec=tkz_dec)


def create_json_field_recall_dataloader(dataset: JsonFieldRecallDataset, batch_size: int):
    return create_recall_dataloader(dataset, batch_size, name='JsonFieldRecall')
