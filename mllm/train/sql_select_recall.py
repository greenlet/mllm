"""SQL selection/aggregate recall dataset (structured-extraction family §2.5).

Automated, grounded in real Wikipedia text: each sample builds a small tabular
context and asks a SQL (or SQL-like NL) query.

- Tier-E: verbatim cell selection (SELECT col FROM t WHERE id = x).
- Tier-C: light aggregates (COUNT/SUM/MAX), derived from rows.

The dataset emits ``MaskedCiteBatch`` and reuses the existing
``MixedDecoder.run_on_text_citation`` path.
"""
from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizer

from mllm.data.utils import TokensSubsetV2
from mllm.train.encdec_graph_bert import MaskedCiteBatch


@dataclass(kw_only=True)
class SqlSelectRecallCfg:
    min_rows: int = 4
    max_rows: int = 8
    min_cols: int = 3
    max_cols: int = 5
    value_min_words: int = 1
    value_max_words: int = 3
    numeric_col_prob: float = 0.45
    # Probability of Tier-C aggregate queries.
    transform_prob: float = 0.30
    min_col_chars: int = 2


class SqlSelectRecallTokenizer:
    def __init__(
            self, tkz_enc: PreTrainedTokenizer, max_len: int, cfg: Optional[SqlSelectRecallCfg] = None,
            n_special_toks: int = 1000, tkz_dec: Optional[PreTrainedTokenizer] = None,
            seed: Optional[int] = None,
    ):
        self.tkz_enc = tkz_enc
        self.tkz_dec = tkz_dec if tkz_dec is not None else tkz_enc
        self.same_vocab = self.tkz_enc is self.tkz_dec or len(self.tkz_enc) == len(self.tkz_dec)
        self.max_len = max_len
        self.cfg = cfg if cfg is not None else SqlSelectRecallCfg()
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

    def _normalize_name(self, txt: str) -> str:
        out = []
        for ch in txt.lower():
            if ch.isalnum() or ch == '_':
                out.append(ch)
            elif ch in ('-', ' '):
                out.append('_')
        name = ''.join(out).strip('_')
        if name and name[0].isdigit():
            name = f'c_{name}'
        return name

    def _is_col_name(self, name: str) -> bool:
        return len(name) >= self.cfg.min_col_chars and any(c.isalpha() for c in name)

    def _pick_col_name(self, pool: List[str], used: set[str]) -> str:
        for _ in range(max(8, len(pool))):
            if pool:
                c = pool[int(self.rng.integers(len(pool)))]
                if c not in used and self._is_col_name(c):
                    used.add(c)
                    return c
            else:
                break
        while True:
            c = f'c_{int(self.rng.integers(10_000_000))}'
            if c not in used:
                used.add(c)
                return c

    def _pick_text_value(self, words: List[List[int]]) -> str:
        for _ in range(10):
            if not words:
                tok = [self._next_pool_token()]
                txt = self.tkz_enc.decode(tok, skip_special_tokens=True).strip()
            else:
                n = int(self.rng.integers(self.cfg.value_min_words, self.cfg.value_max_words + 1))
                i0 = int(self.rng.integers(len(words)))
                ids: List[int] = []
                for w in words[i0:i0 + n]:
                    ids.extend(w)
                txt = self.tkz_enc.decode(ids, skip_special_tokens=True).strip()
            txt = txt.replace('|', ' ').replace('\n', ' ')
            if txt and any(c.isalnum() for c in txt):
                return txt
        return f'v{int(self.rng.integers(100000))}'

    def _build_table(self, words: List[List[int]], col_pool: List[str]) -> Tuple[List[str], List[dict[str, Any]], List[str], List[str]]:
        cfg = self.cfg
        n_rows = int(self.rng.integers(cfg.min_rows, cfg.max_rows + 1))
        n_cols = int(self.rng.integers(cfg.min_cols, cfg.max_cols + 1))

        used = {'id'}
        cols = ['id']
        col_types = ['num']  # id numeric
        while len(cols) < n_cols:
            c = self._pick_col_name(col_pool, used)
            cols.append(c)
            t = 'num' if float(self.rng.random()) < cfg.numeric_col_prob else 'txt'
            col_types.append(t)

        rows: List[dict[str, Any]] = []
        for i in range(n_rows):
            row = {'id': i + 1}
            for c, t in zip(cols[1:], col_types[1:]):
                if t == 'num':
                    row[c] = int(self.rng.integers(0, 1_000_000))
                else:
                    row[c] = self._pick_text_value(words)
            rows.append(row)

        # Markdown-ish table for compact readability.
        header = '| ' + ' | '.join(cols) + ' |'
        sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
        body = []
        for r in rows:
            vals = [str(r[c]) for c in cols]
            body.append('| ' + ' | '.join(vals) + ' |')
        lines = [header, sep] + body
        return cols, rows, col_types, lines

    def _build_query_and_answer(self, cols: List[str], rows: List[dict[str, Any]], col_types: List[str]) -> Tuple[str, str]:
        do_transform = float(self.rng.random()) < self.cfg.transform_prob
        ridx = int(self.rng.integers(len(rows)))
        row = rows[ridx]

        # Tier-E: SELECT col WHERE id = x (verbatim cell answer).
        if not do_transform:
            ci = int(self.rng.integers(1, len(cols)))
            col = cols[ci]
            v = row[col]
            if int(self.rng.integers(2)) == 0:
                q = f'[Tier-E] SQL: SELECT {col} FROM t WHERE id = {row["id"]}; Return value only.'
            else:
                q = f'[Tier-E] Which {col} has id {row["id"]}? Return value only.'
            return q, str(v)

        # Tier-C: COUNT/SUM/MAX
        num_cols = [c for c, t in zip(cols, col_types) if c != 'id' and t == 'num']
        op_choices = ['count']
        if num_cols:
            op_choices += ['sum', 'max']
        op = op_choices[int(self.rng.integers(len(op_choices)))]

        if op == 'count':
            # Random id threshold yields non-trivial counts.
            thr = int(self.rng.integers(1, len(rows) + 1))
            ans = sum(1 for r in rows if r['id'] <= thr)
            q = f'[Tier-C] SQL: SELECT COUNT(*) FROM t WHERE id <= {thr}; Return value only.'
            return q, str(ans)

        col = num_cols[int(self.rng.integers(len(num_cols)))]
        if op == 'sum':
            ans = int(sum(int(r[col]) for r in rows))
            q = f'[Tier-C] SQL: SELECT SUM({col}) FROM t; Return value only.'
            return q, str(ans)

        ans = int(max(int(r[col]) for r in rows))
        q = f'[Tier-C] SQL: SELECT MAX({col}) FROM t; Return value only.'
        return q, str(ans)

    def build(self, text: str) -> TokensSubsetV2:
        ids_src = self.tkz_enc(text, add_special_tokens=False).input_ids
        words = self._split_words(ids_src)

        col_pool: List[str] = []
        seen = set()
        for w in words:
            c = self._normalize_name(self.tkz_enc.decode(w, skip_special_tokens=True).strip())
            if self._is_col_name(c) and c not in seen and c != 'id':
                seen.add(c)
                col_pool.append(c)

        budget = self.max_len - 2
        best = None
        for _ in range(20):
            cols, rows, col_types, lines = self._build_table(words, col_pool)
            table_text = '\n'.join(lines)
            rec_ids = self.tkz_enc(table_text, add_special_tokens=False).input_ids
            if len(rec_ids) <= budget and len(rows) > 0:
                best = (cols, rows, col_types, table_text, rec_ids)
                break
            if best is None or len(rec_ids) < len(best[4]):
                best = (cols, rows, col_types, table_text, rec_ids)

        assert best is not None
        cols, rows, col_types, table_text, rec_ids = best
        if len(rec_ids) > budget:
            rec_ids = rec_ids[:budget]
            table_text = self.tkz_enc.decode(rec_ids, skip_special_tokens=True)

        prompt, target_text = self._build_query_and_answer(cols, rows, col_types)

        prompt_toks = self.tkz_dec(prompt, add_special_tokens=False).input_ids
        target_dec = self.tkz_dec(target_text, add_special_tokens=False).input_ids
        end_suffix = [self.dec_end_token_id] if self.dec_end_token_id is not None else []
        cites_dec = target_dec + end_suffix
        inp_dec = self._enc_to_dec(rec_ids) + end_suffix

        cls_id = self.tkz_enc.cls_token_id
        sep_id = self.tkz_enc.sep_token_id
        toks_inp = [cls_id] + rec_ids + [sep_id]
        tgt_enc = self.tkz_enc(target_text, add_special_tokens=False).input_ids

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
            prompt=prompt,
            toks_prompt=prompt_toks,
            toks_inp_dec=inp_dec,
            toks_inp_masked_dec=list(inp_dec),
            toks_cite_dec=cites_dec,
            toks_cite_masked_dec=list(cites_dec),
        )

    def __call__(self, texts: List[str]) -> List[TokensSubsetV2]:
        return [self.build(t) for t in texts]


class SqlSelectRecallDataset:
    def __init__(
            self, dataset: Dataset, tkz_enc: PreTrainedTokenizer, max_seq_len: int,
            n_special_toks: int = 1000, cfg: Optional[SqlSelectRecallCfg] = None,
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
        self.builder = SqlSelectRecallTokenizer(
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

    def shuffle(self, seed: Optional[int] = None) -> 'SqlSelectRecallDataset':
        if seed is not None:
            np.random.default_rng(seed).shuffle(self.inds)
        else:
            np.random.shuffle(self.inds)
        return self


def create_sql_select_recall_dataloader(
        dataset: SqlSelectRecallDataset, batch_size: int,
) -> Generator[MaskedCiteBatch, None, None]:
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f'R{rank}. Create SqlSelectRecallDataset dataloader. batch_size={batch_size}.')
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
