"""JSONata/jq-style selection and transform dataset (structured-extraction family §2.3).

This generator emits JSON records plus queries in a compact JSONata/jq-like
dialect. It supports:

- Tier-E: pure selection (returns a stored value at a path).
- Tier-C: light transforms over arrays (count/sum/max), where the answer is derived.

Testability split
-----------------
``sample_spec`` fixes the JSON shape, separators, whether the query is a
transform, and the (modular) query / op / dialect indices — all randomness lives
here. :meth:`JsonataRecallTokenizer.realize` deterministically materializes the
record from the word feed and resolves the query/answer.
"""
from dataclasses import dataclass
import json
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

from mllm.data.utils import TokensSubsetV2
from mllm.train.extraction_common import (
    BaseRecallDataset, JsonNodeSpec, WordFeed, assemble_subset, build_json_value,
    build_to_budget, create_recall_dataloader, dec_end_token_id, json_seps,
    json_target_text, make_pool, path_to_dot, path_to_jq, sample_json_shape,
    word_feed_from_text,
)


@dataclass(kw_only=True)
class JsonataRecallCfg:
    min_fields: int = 4
    max_fields: int = 10
    max_depth: int = 3
    max_array_len: int = 5
    value_min_words: int = 1
    value_max_words: int = 3
    nested_prob: float = 0.35
    array_prob: float = 0.28
    number_prob: float = 0.30
    bool_null_prob: float = 0.10
    min_key_chars: int = 2
    # Probability of Tier-C transform samples (count/sum/max).
    transform_prob: float = 0.35
    # Probability of a composite (subtree) Tier-E target.
    composite_prob: float = 0.0
    # When True, grow records to ~fill the chunk (largest sampled record that
    # fits the token budget; early-stop at fill_frac * budget).
    fill_to_budget: bool = False
    fill_frac: float = 0.85


@dataclass(kw_only=True)
class JsonataSpec:
    root: JsonNodeSpec
    compact: bool
    transform: bool       # Tier-C transform vs Tier-E selection
    query_idx: int        # modular index into leaves (Tier-E) or arrays (Tier-C)
    op_idx: int           # modular index into available transform ops
    dialect: int          # transform dialect: 0 jsonata / 1 jq
    e_dialect: int        # selection dialect: 0 jsonata / 1 jq
    composite: bool = False


class JsonataRecallTokenizer:
    def __init__(
            self, tkz_enc: PreTrainedTokenizer, max_len: int, cfg: Optional[JsonataRecallCfg] = None,
            n_special_toks: int = 1000, tkz_dec: Optional[PreTrainedTokenizer] = None,
            seed: Optional[int] = None,
    ):
        self.tkz_enc = tkz_enc
        self.tkz_dec = tkz_dec if tkz_dec is not None else tkz_enc
        self.same_vocab = self.tkz_enc is self.tkz_dec or len(self.tkz_enc) == len(self.tkz_dec)
        self.max_len = max_len
        self.cfg = cfg if cfg is not None else JsonataRecallCfg()
        self.rng = np.random.default_rng(seed)
        self.pool = make_pool(self.rng, n_special_toks, len(self.tkz_enc))
        self.dec_end_token_id = dec_end_token_id(self.tkz_dec)

    # --- random: spec sampling ----------------------------------------------
    def sample_spec(self) -> JsonataSpec:
        cfg = self.cfg
        rng = self.rng
        return JsonataSpec(
            root=sample_json_shape(rng, cfg),
            compact=bool(rng.integers(2)),
            transform=float(rng.random()) < cfg.transform_prob,
            query_idx=int(rng.integers(1 << 30)),
            op_idx=int(rng.integers(1 << 30)),
            dialect=int(rng.integers(2)),
            e_dialect=int(rng.integers(2)),
            composite=float(rng.random()) < cfg.composite_prob,
        )

    # --- deterministic: transform query -------------------------------------
    def _transform_query(self, spec: JsonataSpec, arrays: List[Tuple[List, list]]) -> Optional[Tuple[str, Any]]:
        if not arrays:
            return None
        array_path, arr = arrays[spec.query_idx % len(arrays)]
        dot = path_to_dot(array_path)
        jq = path_to_jq(array_path)
        numeric = [x for x in arr if isinstance(x, (int, float)) and not isinstance(x, bool)]
        ops = ['count'] + (['sum', 'max'] if numeric else [])
        op = ops[spec.op_idx % len(ops)]
        if op == 'count':
            ans = len(arr)
            q = (f'[Tier-C] JSONata: $count({dot}). Return value only.'
                 if spec.dialect == 0 else f'[Tier-C] jq: {jq} | length. Return value only.')
            return q, ans
        if op == 'sum':
            ans = int(np.sum(np.array(numeric, dtype=np.int64)))
            q = (f'[Tier-C] JSONata: $sum({dot}). Return value only.'
                 if spec.dialect == 0 else f'[Tier-C] jq: {jq} | add. Return value only.')
            return q, ans
        ans = int(np.max(np.array(numeric, dtype=np.int64)))
        q = (f'[Tier-C] JSONata: $max({dot}). Return value only.'
             if spec.dialect == 0 else f'[Tier-C] jq: {jq} | max. Return value only.')
        return q, ans

    # --- deterministic: realization -----------------------------------------
    def realize(self, spec: JsonataSpec, feed: WordFeed, ids_src: Optional[List[int]] = None) -> Tuple[TokensSubsetV2, int]:
        budget = self.max_len - 2
        obj, leaves, arrays, composites = build_json_value(spec.root, feed, self.cfg.min_key_chars)

        seps = json_seps(spec.compact)
        rec_text = json.dumps(obj, ensure_ascii=True, separators=seps)
        rec_ids = self.tkz_enc(rec_text, add_special_tokens=False).input_ids
        rec_len = len(rec_ids)
        if rec_len > budget:
            rec_ids = rec_ids[:budget]
            rec_text = self.tkz_enc.decode(rec_ids, skip_special_tokens=True)

        query: Optional[str] = None
        answer_value: Any = None
        if spec.transform:
            tr = self._transform_query(spec, arrays)
            if tr is not None:
                query, answer_value = tr

        if query is None:
            candidates = composites if (spec.composite and composites) else leaves
            if not candidates:
                candidates = leaves or [([], obj)]
            n = len(candidates)
            path, value = candidates[spec.query_idx % n]
            tgt = json_target_text(value, spec.compact)
            for j in range(n):
                if tgt and tgt in rec_text:
                    break
                path, value = candidates[(spec.query_idx + j + 1) % n]
                tgt = json_target_text(value, spec.compact)
            dot = path_to_dot(path)
            jq = path_to_jq(path)
            query = (f'[Tier-E] JSONata: {dot}. Return value only.'
                     if spec.e_dialect == 0 else f'[Tier-E] jq: {jq}. Return value only.')
            answer_value = value

        tgt_text = json_target_text(answer_value, spec.compact)
        prompt_toks = self.tkz_dec(query, add_special_tokens=False).input_ids

        subset = assemble_subset(
            ids_src=ids_src if ids_src is not None else list(rec_ids),
            rec_ids=rec_ids,
            prompt=query,
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
        feed, ids_src = word_feed_from_text(text, self.tkz_enc, self.pool)
        return build_to_budget(
            sample_spec=self.sample_spec,
            realize=self.realize,
            feed=feed,
            ids_src=ids_src,
            budget=self.max_len - 2,
            n_attempts=20,
            fill_to_budget=self.cfg.fill_to_budget,
            fill_frac=self.cfg.fill_frac,
        )

    def __call__(self, texts: List[str]) -> List[TokensSubsetV2]:
        return [self.build(t) for t in texts]


class JsonataRecallDataset(BaseRecallDataset):
    def __init__(
            self, dataset: Dataset, tkz_enc: PreTrainedTokenizer, max_seq_len: int,
            n_special_toks: int = 1000, cfg: Optional[JsonataRecallCfg] = None,
            device: Optional[torch.device] = None, tkz_dec: Optional[PreTrainedTokenizer] = None,
            seed: Optional[int] = None,
    ):
        builder = JsonataRecallTokenizer(
            tkz_enc, max_len=max_seq_len, cfg=cfg, n_special_toks=n_special_toks,
            tkz_dec=tkz_dec if tkz_dec is not None else tkz_enc, seed=seed,
        )
        super().__init__(dataset, tkz_enc, max_seq_len, builder, device=device, tkz_dec=tkz_dec)


def create_jsonata_recall_dataloader(dataset: JsonataRecallDataset, batch_size: int):
    return create_recall_dataloader(dataset, batch_size, name='JsonataRecall')
