"""JSONata/jq-style selection and transform dataset (structured-extraction family §2.3).

This generator emits JSON records plus queries in a compact JSONata/jq-like
dialect. It supports:

- Tier-E: pure selection (returns a stored scalar at a path).
- Tier-C: light transforms over arrays (count/sum/max), where the answer is derived.

The output shape is compatible with ``MaskedCiteBatch`` and routes through
``MixedDecoder.run_on_text_citation`` unchanged.
"""
from dataclasses import dataclass
import json
from typing import Any, Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizer

from mllm.data.utils import TokensSubsetV2
from mllm.train.encdec_graph_bert import MaskedCiteBatch


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
        self.pool = self.rng.permutation(np.arange(n_special_toks, len(self.tkz_enc)))
        self.pool_cur = 0
        td = self.tkz_dec
        self.dec_end_token_id = td.sep_token_id if td.sep_token_id is not None else td.eos_token_id

    def _next_pool_token(self) -> int:
        tid = int(self.pool[self.pool_cur])
        self.pool_cur += 1
        if self.pool_cur >= len(self.pool):
            self.rng.shuffle(self.pool)
            self.pool_cur = 0
        return tid

    def _enc_to_dec(self, enc_ids: List[int]) -> List[int]:
        if self.same_vocab:
            return list(enc_ids)
        if not enc_ids:
            return []
        text = self.tkz_enc.decode(enc_ids, skip_special_tokens=True)
        return self.tkz_dec(text, add_special_tokens=False).input_ids

    def _split_words(self, ids: List[int]) -> List[List[int]]:
        toks = self.tkz_enc.convert_ids_to_tokens(ids)
        words: List[List[int]] = []
        cur: List[int] = []
        for tid, tok in zip(ids, toks):
            if tok.startswith('##') and cur:
                cur.append(tid)
            else:
                if cur:
                    words.append(cur)
                cur = [tid]
        if cur:
            words.append(cur)
        return words

    def _normalize_key(self, txt: str) -> str:
        out = []
        for ch in txt.lower():
            if ch.isalnum() or ch == '_':
                out.append(ch)
            elif ch in ('-', ' '):
                out.append('_')
        return ''.join(out).strip('_')

    def _is_key(self, key: str) -> bool:
        return len(key) >= self.cfg.min_key_chars and any(c.isalpha() for c in key)

    def _pick_key(self, key_pool: List[str], used: set[str]) -> str:
        for _ in range(max(8, len(key_pool))):
            if key_pool:
                k = key_pool[int(self.rng.integers(len(key_pool)))]
                if k not in used and self._is_key(k):
                    used.add(k)
                    return k
            else:
                break
        while True:
            k = f'k_{int(self.rng.integers(10_000_000))}'
            if k not in used:
                used.add(k)
                return k

    def _pick_text_value(self, words: List[List[int]]) -> str:
        if not words:
            tok = [self._next_pool_token()]
            txt = self.tkz_enc.decode(tok, skip_special_tokens=True).strip()
            return txt if txt else f'v{int(self.rng.integers(100000))}'
        n = int(self.rng.integers(self.cfg.value_min_words, self.cfg.value_max_words + 1))
        i0 = int(self.rng.integers(len(words)))
        v_ids: List[int] = []
        for w in words[i0:i0 + n]:
            v_ids.extend(w)
        txt = self.tkz_enc.decode(v_ids, skip_special_tokens=True).strip()
        return txt if txt else f'v{int(self.rng.integers(100000))}'

    def _pick_scalar(self, words: List[List[int]]) -> Any:
        r = float(self.rng.random())
        if r < self.cfg.bool_null_prob * 0.5:
            return bool(self.rng.integers(2))
        if r < self.cfg.bool_null_prob:
            return None
        if r < self.cfg.bool_null_prob + self.cfg.number_prob:
            return int(self.rng.integers(0, 1_000_000))
        return self._pick_text_value(words)

    def _target_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=True)

    def _path_to_dot(self, path: List[Any]) -> str:
        parts: List[str] = []
        for p in path:
            if isinstance(p, int):
                if parts:
                    parts[-1] = f'{parts[-1]}[{p}]'
                else:
                    parts.append(f'[{p}]')
            else:
                parts.append(str(p))
        return '.'.join(parts)

    def _path_to_jq(self, path: List[Any]) -> str:
        out = '.'
        for p in path:
            if isinstance(p, int):
                out += f'[{p}]'
            else:
                out += p if out == '.' else f'.{p}'
        return out

    def _make_object(
            self, words: List[List[int]], key_pool: List[str],
    ) -> Tuple[dict[str, Any], List[Tuple[List[Any], Any]], List[Tuple[List[Any], List[Any]]]]:
        cfg = self.cfg
        n_fields = int(self.rng.integers(cfg.min_fields, cfg.max_fields + 1))
        root: dict[str, Any] = {}
        leaves: List[Tuple[List[Any], Any]] = []
        # (array_path, array_values)
        arrays: List[Tuple[List[Any], List[Any]]] = []
        used_root: set[str] = set()

        def fill_dict(dst: dict[str, Any], depth: int, path_prefix: List[Any], used_here: set[str]):
            nonlocal n_fields
            while n_fields > 0:
                key = self._pick_key(key_pool, used_here)
                n_fields -= 1
                r = float(self.rng.random())
                can_nest = depth < cfg.max_depth and n_fields > 0
                if can_nest and r < cfg.nested_prob:
                    child: dict[str, Any] = {}
                    dst[key] = child
                    fill_dict(child, depth + 1, path_prefix + [key], set())
                elif r < cfg.nested_prob + cfg.array_prob:
                    arr_len = int(self.rng.integers(1, cfg.max_array_len + 1))
                    arr = [self._pick_scalar(words) for _ in range(arr_len)]
                    dst[key] = arr
                    arrays.append((path_prefix + [key], arr))
                    qi = int(self.rng.integers(arr_len))
                    leaves.append((path_prefix + [key, qi], arr[qi]))
                else:
                    val = self._pick_scalar(words)
                    dst[key] = val
                    leaves.append((path_prefix + [key], val))
                if depth > 0 and float(self.rng.random()) < 0.35:
                    break

        fill_dict(root, 1, [], used_root)
        if not leaves:
            k = self._pick_key(key_pool, used_root)
            v = self._pick_scalar(words)
            root[k] = v
            leaves.append(([k], v))
        return root, leaves, arrays

    def _build_transform_query(self, arrays: List[Tuple[List[Any], List[Any]]]) -> Optional[Tuple[str, str, Any]]:
        """Return (tier_tag, query, answer_value) or None if no valid transform target."""
        if not arrays:
            return None
        # 1) Count always valid.
        # 2) Sum/Max only for arrays with at least one numeric value.
        array_path, arr = arrays[int(self.rng.integers(len(arrays)))]
        dot = self._path_to_dot(array_path)
        jq = self._path_to_jq(array_path)

        numeric = [x for x in arr if isinstance(x, (int, float)) and not isinstance(x, bool)]
        ops = ['count']
        if numeric:
            ops.extend(['sum', 'max'])
        op = ops[int(self.rng.integers(len(ops)))]
        dialect = int(self.rng.integers(2))  # 0=jsonata, 1=jq

        if op == 'count':
            ans = len(arr)
            if dialect == 0:
                q = f'[Tier-C] JSONata: $count({dot}). Return value only.'
            else:
                q = f'[Tier-C] jq: {jq} | length. Return value only.'
            return 'C', q, ans

        if op == 'sum':
            ans = int(np.sum(np.array(numeric, dtype=np.int64)))
            if dialect == 0:
                q = f'[Tier-C] JSONata: $sum({dot}). Return value only.'
            else:
                q = f'[Tier-C] jq: {jq} | add. Return value only.'
            return 'C', q, ans

        ans = int(np.max(np.array(numeric, dtype=np.int64)))
        if dialect == 0:
            q = f'[Tier-C] JSONata: $max({dot}). Return value only.'
        else:
            q = f'[Tier-C] jq: {jq} | max. Return value only.'
        return 'C', q, ans

    def build(self, text: str) -> TokensSubsetV2:
        ids_src = self.tkz_enc(text, add_special_tokens=False).input_ids
        words = self._split_words(ids_src)

        key_pool: List[str] = []
        seen = set()
        for w in words:
            k = self._normalize_key(self.tkz_enc.decode(w, skip_special_tokens=True).strip())
            if self._is_key(k) and k not in seen:
                seen.add(k)
                key_pool.append(k)

        budget = self.max_len - 2
        best = None
        for _ in range(20):
            obj, leaves, arrays = self._make_object(words, key_pool)
            compact = bool(self.rng.integers(2))
            seps = (',', ':') if compact else (', ', ': ')
            rec_text = json.dumps(obj, ensure_ascii=True, separators=seps)
            rec_ids = self.tkz_enc(rec_text, add_special_tokens=False).input_ids
            if len(rec_ids) <= budget and leaves:
                best = (obj, leaves, arrays, rec_text, rec_ids)
                break
            if best is None or len(rec_ids) < len(best[4]):
                best = (obj, leaves, arrays, rec_text, rec_ids)

        assert best is not None
        _, leaves, arrays, rec_text, rec_ids = best
        if len(rec_ids) > budget:
            rec_ids = rec_ids[:budget]
            rec_text = self.tkz_enc.decode(rec_ids, skip_special_tokens=True)

        # Choose Tier-C transform vs Tier-E selection.
        query = None
        answer_value: Any = None
        do_transform = float(self.rng.random()) < self.cfg.transform_prob
        if do_transform:
            tr = self._build_transform_query(arrays)
            if tr is not None:
                _, query, answer_value = tr

        if query is None:
            path, value = leaves[int(self.rng.integers(len(leaves)))]
            tgt = self._target_text(value)
            for _ in range(12):
                if tgt and tgt in rec_text:
                    break
                path, value = leaves[int(self.rng.integers(len(leaves)))]
                tgt = self._target_text(value)
            dot = self._path_to_dot(path)
            jq = self._path_to_jq(path)
            if int(self.rng.integers(2)) == 0:
                query = f'[Tier-E] JSONata: {dot}. Return value only.'
            else:
                query = f'[Tier-E] jq: {jq}. Return value only.'
            answer_value = value

        tgt_text = self._target_text(answer_value)

        prompt_toks = self.tkz_dec(query, add_special_tokens=False).input_ids
        target_dec = self.tkz_dec(tgt_text, add_special_tokens=False).input_ids
        end_suffix = [self.dec_end_token_id] if self.dec_end_token_id is not None else []
        cites_dec = target_dec + end_suffix
        inp_dec = self._enc_to_dec(rec_ids) + end_suffix

        cls_id = self.tkz_enc.cls_token_id
        sep_id = self.tkz_enc.sep_token_id
        toks_inp = [cls_id] + rec_ids + [sep_id]

        tgt_enc = self.tkz_enc(tgt_text, add_special_tokens=False).input_ids

        return TokensSubsetV2(
            toks_src=ids_src,
            inp_beg_ind=0,
            inp_end_ind=len(ids_src),
            toks_inp=toks_inp,
            toks_inp_masked=list(toks_inp),
            cite_beg_ind=-1,
            cite_end_ind=-1,
            toks_cite=tgt_enc,
            toks_cite_masked=list(tgt_enc),
            toks_cite_beg=[],
            toks_cite_end=[],
            prompt=query,
            toks_prompt=prompt_toks,
            toks_inp_dec=inp_dec,
            toks_inp_masked_dec=list(inp_dec),
            toks_cite_dec=cites_dec,
            toks_cite_masked_dec=list(cites_dec),
        )

    def __call__(self, texts: List[str]) -> List[TokensSubsetV2]:
        return [self.build(t) for t in texts]


class JsonataRecallDataset:
    def __init__(
            self, dataset: Dataset, tkz_enc: PreTrainedTokenizer, max_seq_len: int,
            n_special_toks: int = 1000, cfg: Optional[JsonataRecallCfg] = None,
            device: Optional[torch.device] = None, tkz_dec: Optional[PreTrainedTokenizer] = None,
            seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.tkz_enc = tkz_enc
        self.tkz_dec = tkz_dec if tkz_dec is not None else tkz_enc
        self.size = len(dataset)
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device('cpu')
        self.enc_pad_token_id = tkz_enc.pad_token_id
        self.dec_pad_token_id = self.tkz_dec.pad_token_id
        if self.dec_pad_token_id is None:
            self.dec_pad_token_id = self.tkz_dec.eos_token_id
        self.inds = np.arange(self.size)
        self.builder = JsonataRecallTokenizer(
            tkz_enc, max_len=max_seq_len, cfg=cfg, n_special_toks=n_special_toks,
            tkz_dec=self.tkz_dec, seed=seed,
        )

    def __len__(self):
        return self.size

    def _to_tensor(self, toks_list: List[List[int]], pad_token_id: int, with_att_mask: bool):
        batch_size = len(toks_list)
        max_len = max((len(t) for t in toks_list), default=1)
        max_len = max(max_len, 1)
        out = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=self.device)
        att = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device) if with_att_mask else None
        for i, toks in enumerate(toks_list):
            n = len(toks)
            if n:
                out[i, :n] = torch.tensor(toks, dtype=torch.long, device=self.device)
            if att is not None:
                att[i, :n] = 1
        return out, att

    def get_batch(self, inds: List[int]) -> MaskedCiteBatch:
        texts = [self.dataset[int(i)]['text'] for i in inds]
        subsets = self.builder(texts)
        batch_size = len(inds)
        enc_pad, dec_pad = self.enc_pad_token_id, self.dec_pad_token_id

        inp_toks, inp_att_mask = self._to_tensor([s.toks_inp for s in subsets], enc_pad, True)
        inp_masked_toks, _ = self._to_tensor([s.toks_inp_masked for s in subsets], enc_pad, False)
        prompts_toks, prompts_att_mask = self._to_tensor([s.toks_prompt for s in subsets], dec_pad, True)
        cites_toks, cites_att_mask = self._to_tensor([s.toks_cite_dec for s in subsets], dec_pad, True)
        cites_masked_toks, _ = self._to_tensor([s.toks_cite_masked_dec for s in subsets], dec_pad, False)
        inp_toks_dec, inp_dec_att_mask = self._to_tensor([s.toks_inp_dec for s in subsets], dec_pad, True)
        inp_masked_toks_dec, _ = self._to_tensor([s.toks_inp_masked_dec for s in subsets], dec_pad, False)
        edge_inds = torch.stack([
            torch.arange(batch_size, device=self.device),
            torch.full((batch_size,), batch_size, device=self.device),
        ])

        return MaskedCiteBatch(
            tokens_subsets=subsets,
            inp_toks=inp_toks,
            inp_masked_toks=inp_masked_toks,
            prompts_toks=prompts_toks,
            cites_masked_toks=cites_masked_toks,
            cites_toks=cites_toks,
            inp_att_mask=inp_att_mask,
            prompts_att_mask=prompts_att_mask,
            cites_att_mask=cites_att_mask,
            inp_toks_dec=inp_toks_dec,
            inp_masked_toks_dec=inp_masked_toks_dec,
            inp_dec_att_mask=inp_dec_att_mask,
            edge_inds=edge_inds,
        )

    def shuffle(self, seed: Optional[int] = None) -> 'JsonataRecallDataset':
        if seed is not None:
            np.random.default_rng(seed).shuffle(self.inds)
        else:
            np.random.shuffle(self.inds)
        return self


def create_jsonata_recall_dataloader(
        dataset: JsonataRecallDataset, batch_size: int,
) -> Generator[MaskedCiteBatch, None, None]:
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f'R{rank}. Create JsonataRecallDataset dataloader. batch_size={batch_size}.')
    start_ind = 0
    while True:
        end_ind = min(start_ind + batch_size, len(dataset))
        inds = dataset.inds[start_ind:end_ind].tolist()
        if len(inds) < batch_size:
            inds += dataset.inds[:(batch_size - len(inds))].tolist()
        batch = dataset.get_batch(inds)
        if end_ind == len(dataset):
            print(f'R{rank}. Shuffle dataset')
            dataset.shuffle()
        yield batch
        start_ind = end_ind % len(dataset)
