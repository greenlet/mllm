"""SQL selection/aggregate recall dataset (structured-extraction family §2.5).

Automated, grounded in real Wikipedia text: each sample builds a small tabular
context and asks a SQL (or SQL-like NL) query.

- Tier-E: verbatim cell selection (SELECT col FROM t WHERE id = x).
- Tier-C: light aggregates (COUNT/SUM/MAX), derived from rows.
- Composite: SELECT * (whole row), serialized with the table's ``" | "`` cell
  separator so the answer stays a verbatim substring.

Testability split
-----------------
``sample_spec`` fixes the table *shape* (column types, per-cell literals / word
counts) and the query indices — all randomness lives here.
:meth:`SqlSelectRecallTokenizer.realize` deterministically fills column names and
text cells *sequentially* from the word feed.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

from mllm.data.utils import TokensSubsetV2
from mllm.train.extraction_common import (
    BaseRecallDataset, WordFeed, assemble_subset, build_to_budget,
    create_recall_dataloader, dec_end_token_id, make_pool, word_feed_from_text,
)


@dataclass(kw_only=True)
class SqlSelectRecallCfg:
    min_rows: int = 4
    max_rows: int = 8
    min_cols: int = 3
    max_cols: int = 5
    value_min_words: int = 1
    value_max_words: int = 3
    # Column-name length in *words*. Multi-word names stay distinctive so a
    # frequent stopword cannot collide across columns / queries.
    name_min_words: int = 3
    name_max_words: int = 4
    numeric_col_prob: float = 0.45
    # Probability of Tier-C aggregate queries.
    transform_prob: float = 0.30
    min_col_chars: int = 2
    # Probability of a composite (whole-row SELECT *) target.
    composite_prob: float = 0.0
    # When True, grow tables to ~fill the chunk (largest sampled record that
    # fits the token budget; early-stop at fill_frac * budget).
    fill_to_budget: bool = False
    fill_frac: float = 0.85


@dataclass(kw_only=True)
class SqlCellSpec:
    kind: str            # 'num' | 'txt'
    int_lit: int = 0     # kind == 'num'
    word_count: int = 0  # kind == 'txt'


@dataclass(kw_only=True)
class SqlSpec:
    col_types: List[str]              # col_types[0] == 'num' (the id column)
    rows: List[List[SqlCellSpec]]     # per row: cells for cols[1:]
    transform: bool
    name_words: int                   # words per (non-id) column name
    query_row_idx: int
    query_col_idx: int                # modular into cols[1:]
    e_phrasing: int                   # Tier-E phrasing: 0 SQL / 1 NL
    op_idx: int                       # modular into available Tier-C ops
    count_thr_raw: int                # modular into row count for COUNT threshold
    num_col_idx: int                  # modular into numeric columns
    composite: bool = False


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
        self.pool = make_pool(self.rng, n_special_toks, len(self.tkz_enc))
        self.dec_end_token_id = dec_end_token_id(self.tkz_dec)

    # --- helpers -------------------------------------------------------------
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

    def _next_col_name(self, feed: WordFeed, used: set, n_words: int = 1) -> str:
        parts: List[str] = []
        for _ in range(max(1, n_words)):
            part = None
            for _ in range(64):
                _, txt = feed.next_word()
                c = self._normalize_name(txt)
                if self._is_col_name(c):
                    part = c
                    break
            if part is None:
                for _ in range(256):
                    tid = feed.next_pool_token()
                    c = self._normalize_name(self.tkz_enc.decode([tid], skip_special_tokens=True))
                    if self._is_col_name(c):
                        part = c
                        break
                if part is None:
                    part = f'c_{len(used)}'
            parts.append(part)
        name = '_'.join(parts)
        if name in used:
            suffix = 1
            while f'{name}_{suffix}' in used:
                suffix += 1
            name = f'{name}_{suffix}'
        used.add(name)
        return name

    # --- random: spec sampling ----------------------------------------------
    def sample_spec(self) -> SqlSpec:
        cfg = self.cfg
        rng = self.rng
        n_rows = int(rng.integers(cfg.min_rows, cfg.max_rows + 1))
        n_cols = int(rng.integers(cfg.min_cols, cfg.max_cols + 1))
        col_types = ['num']  # id column
        for _ in range(n_cols - 1):
            col_types.append('num' if float(rng.random()) < cfg.numeric_col_prob else 'txt')
        rows: List[List[SqlCellSpec]] = []
        for _ in range(n_rows):
            row: List[SqlCellSpec] = []
            for t in col_types[1:]:
                if t == 'num':
                    row.append(SqlCellSpec(kind='num', int_lit=int(rng.integers(0, 1_000_000))))
                else:
                    row.append(SqlCellSpec(
                        kind='txt', word_count=int(rng.integers(cfg.value_min_words, cfg.value_max_words + 1)),
                    ))
            rows.append(row)
        return SqlSpec(
            col_types=col_types,
            rows=rows,
            transform=float(rng.random()) < cfg.transform_prob,
            name_words=int(rng.integers(cfg.name_min_words, cfg.name_max_words + 1)),
            query_row_idx=int(rng.integers(1 << 30)),
            query_col_idx=int(rng.integers(1 << 30)),
            e_phrasing=int(rng.integers(2)),
            op_idx=int(rng.integers(1 << 30)),
            count_thr_raw=int(rng.integers(1 << 30)),
            num_col_idx=int(rng.integers(1 << 30)),
            composite=float(rng.random()) < cfg.composite_prob,
        )

    # --- deterministic: realization -----------------------------------------
    def _resolve_query(self, spec: SqlSpec, cols: List[str], rows: List[dict]) -> Tuple[str, str]:
        n_cols = len(cols)
        ridx = spec.query_row_idx % len(rows)
        row = rows[ridx]

        if not spec.transform:
            if spec.composite:
                cells = ' | '.join(str(row[c]) for c in cols)
                q = f'[Tier-E] SQL: SELECT * FROM t WHERE id = {row["id"]}; Return row only.'
                return q, cells
            ci = 1 + (spec.query_col_idx % (n_cols - 1))
            col = cols[ci]
            v = row[col]
            if spec.e_phrasing == 0:
                q = f'[Tier-E] SQL: SELECT {col} FROM t WHERE id = {row["id"]}; Return value only.'
            else:
                q = f'[Tier-E] Which {col} has id {row["id"]}? Return value only.'
            return q, str(v)

        # Tier-C aggregate
        num_cols = [c for c, t in zip(cols, spec.col_types) if c != 'id' and t == 'num']
        ops = ['count'] + (['sum', 'max'] if num_cols else [])
        op = ops[spec.op_idx % len(ops)]
        if op == 'count':
            thr = 1 + (spec.count_thr_raw % len(rows))
            ans = sum(1 for r in rows if r['id'] <= thr)
            return f'[Tier-C] SQL: SELECT COUNT(*) FROM t WHERE id <= {thr}; Return value only.', str(ans)
        col = num_cols[spec.num_col_idx % len(num_cols)]
        if op == 'sum':
            ans = int(sum(int(r[col]) for r in rows))
            return f'[Tier-C] SQL: SELECT SUM({col}) FROM t; Return value only.', str(ans)
        ans = int(max(int(r[col]) for r in rows))
        return f'[Tier-C] SQL: SELECT MAX({col}) FROM t; Return value only.', str(ans)

    def realize(self, spec: SqlSpec, feed: WordFeed, ids_src: Optional[List[int]] = None) -> Tuple[TokensSubsetV2, int]:
        budget = self.max_len - 2

        used = {'id'}
        cols = ['id']
        for _ in range(len(spec.col_types) - 1):
            cols.append(self._next_col_name(feed, used, spec.name_words))

        rows: List[dict] = []
        for i, rowspec in enumerate(spec.rows):
            row: dict = {'id': i + 1}
            for c, cell in zip(cols[1:], rowspec):
                if cell.kind == 'num':
                    row[c] = cell.int_lit
                else:
                    _, v = feed.next_span(cell.word_count)
                    row[c] = v.replace('|', ' ').replace('\n', ' ')
            rows.append(row)

        header = '| ' + ' | '.join(cols) + ' |'
        sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
        body = ['| ' + ' | '.join(str(r[c]) for c in cols) + ' |' for r in rows]
        table_text = '\n'.join([header, sep] + body)

        rec_ids = self.tkz_enc(table_text, add_special_tokens=False).input_ids
        rec_len = len(rec_ids)
        if rec_len > budget:
            rec_ids = rec_ids[:budget]
            table_text = self.tkz_enc.decode(rec_ids, skip_special_tokens=True)

        prompt, target_text = self._resolve_query(spec, cols, rows)
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
            target_text=target_text,
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
            n_attempts=20,
            fill_to_budget=self.cfg.fill_to_budget,
            fill_frac=self.cfg.fill_frac,
        )

    def __call__(self, texts: List[str]) -> List[TokensSubsetV2]:
        return [self.build(t) for t in texts]


class SqlSelectRecallDataset(BaseRecallDataset):
    def __init__(
            self, dataset: Dataset, tkz_enc: PreTrainedTokenizer, max_seq_len: int,
            n_special_toks: int = 1000, cfg: Optional[SqlSelectRecallCfg] = None,
            device: Optional[torch.device] = None, tkz_dec: Optional[PreTrainedTokenizer] = None,
            seed: Optional[int] = None,
    ):
        builder = SqlSelectRecallTokenizer(
            tkz_enc, max_len=max_seq_len, cfg=cfg, n_special_toks=n_special_toks,
            tkz_dec=tkz_dec if tkz_dec is not None else tkz_enc, seed=seed,
        )
        super().__init__(dataset, tkz_enc, max_seq_len, builder, device=device, tkz_dec=tkz_dec)


def create_sql_select_recall_dataloader(dataset: SqlSelectRecallDataset, batch_size: int):
    return create_recall_dataloader(dataset, batch_size, name='SqlSelectRecall')
