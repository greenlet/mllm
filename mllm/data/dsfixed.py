import itertools as it
import os.path
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import trange

from mllm.tokenization.chunk_tokenizer import split_doc_embs, parse_out_subdir, gen_ds_fnames


CACHE_SUBDIR = '.mllm'
CACHE_DF_FNAME = 'ds.csv'


def read_ds_files(ds_dir_path: Path, n_total: int = 0) -> pd.DataFrame:
    cache_fpath = ds_dir_path / CACHE_SUBDIR / CACHE_DF_FNAME
    if cache_fpath.exists():
        print(f'Loading cache from {cache_fpath}')
        nrows = None if n_total <= 0 else n_total
        df = pd.read_csv(cache_fpath, header=0, nrows=nrows)
        print(f'Loaded dataset size: {len(df)}')
        return df
    dfs = []
    fpaths = []
    for i, p in enumerate(ds_dir_path.iterdir()):
        if p.suffix != '.csv':
            continue
        fpaths.append(p)
    n_files = len(fpaths)
    for i in trange(n_files, desc='Processing csv files', unit='file'):
        fpath = fpaths[i]
        df = pd.read_csv(fpath, header=0)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df.sort_values(['docid', 'offset'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.reset_index(drop=False, names='chid', inplace=True)

    print(f'Caching df with {len(df)} items to {cache_fpath}')
    shutil.rmtree(cache_fpath.parent, ignore_errors=True)
    cache_fpath.parent.mkdir()
    df.to_csv(cache_fpath, header=True)
    return df


class DocsBatch:
    docs_chunks: dict[int, list[np.ndarray]]
    target_doc_id: int
    target_tokens: list[int]
    pad_tok: int
    emb_chunk_size: int
    device: torch.device
    docs_chunks_padded: np.ndarray
    target_chunks_padded: np.ndarray
    target_mask: np.ndarray
    fixed_size: bool
    docs_chunks_padded_tf: Optional[torch.Tensor] = None
    target_chunks_padded_tf: Optional[torch.Tensor] = None
    target_mask_tf: Optional[torch.Tensor] = None
    device: Optional[torch.device] = None

    def __init__(self, docs_chunks: dict[int, list[np.ndarray]], target_doc_id: int, target_tokens: list[int],
                 pad_tok: int, emb_chunk_size: int, fixed_size: bool, device: Optional[torch.device] = None):
        self.docs_chunks = docs_chunks
        self.target_doc_id = target_doc_id
        self.target_tokens = target_tokens
        self.pad_tok = pad_tok
        self.emb_chunk_size = emb_chunk_size
        self.fixed_size = fixed_size
        self.device = device
        self.calc_np()

    def calc_np(self):
        docs_chunks = []
        target_chunk_off, target_chunk_sz = 0, 0
        for doc_id, chunks in self.docs_chunks.items():
            if target_chunk_sz == 0:
                if doc_id == self.target_doc_id:
                    target_chunk_sz = len(chunks)
                else:
                    target_chunk_off += len(chunks)
            docs_chunks.extend(chunks)

        target_embs_offsets = split_doc_embs(len(self.target_tokens), self.emb_chunk_size, self.fixed_size)
        n_target_chunks = len(target_embs_offsets) - 1
        target_chunks = []
        for i in range(n_target_chunks):
            chunk = self.target_tokens[target_embs_offsets[i]:target_embs_offsets[i + 1]]
            target_chunks.append(chunk)

        n_batch_chunks = len(docs_chunks)
        max_chank_sz = max(len(chunk) for chunk in it.chain(docs_chunks, target_chunks))

        docs_chunks_padded = np.full((n_batch_chunks, max_chank_sz), self.pad_tok, dtype=np.int32)
        for i_chunk, chunk in enumerate(docs_chunks):
            docs_chunks_padded[i_chunk, :len(chunk)] = chunk

        target_chunks_padded = np.full((n_target_chunks, max_chank_sz), self.pad_tok, dtype=np.int32)
        for i_chunk, chunk in enumerate(target_chunks):
            target_chunks_padded[i_chunk, :len(chunk)] = chunk

        target_mask = np.full(len(docs_chunks), False, dtype=bool)
        target_mask[target_chunk_off:target_chunk_off + target_chunk_sz] = True
        # print(f'target_chunk_off = {target_chunk_off}. target_chunk_sz = {target_chunk_sz}')
        # print(f'target_mask = {target_mask}')

        self.docs_chunks_padded = docs_chunks_padded
        self.target_chunks_padded = target_chunks_padded
        self.target_mask = target_mask

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        res = torch.from_numpy(arr)
        if self.device is not None:
            res = res.to(self.device)
        return res

    def gen_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.docs_chunks_padded_tf is None:
            self.docs_chunks_padded_tf, self.target_chunks_padded_tf, self.target_mask_tf = \
                map(self._to_tensor, (self.docs_chunks_padded, self.target_chunks_padded, self.target_mask))
        return self.docs_chunks_padded_tf, self.target_chunks_padded_tf, self.target_mask_tf


class DsLoader:
    ds_dir_path: Path
    emb_chunk_size: int
    fixed_size: bool
    docs_batch_size: int
    max_chunks_per_doc: int
    pad_tok: int
    qbeg_tok: int
    qend_tok: int
    df: pd.DataFrame
    df_doc: pd.DataFrame
    val_ratio: float
    n_docs: int
    n_docs_train: int
    n_docs_val: int
    docids: np.ndarray
    docids_train: np.ndarray
    docids_val: np.ndarray
    _tokens_cache: dict[tuple[int, int], np.ndarray]
    _max_cache_size: int = 3
    device: Optional[torch.device] = None
    n_total: int = 0

    def __init__(self, ds_dir_path: Path, docs_batch_size: int, max_chunks_per_doc: int,
                 pad_tok: int, qbeg_tok: int, qend_tok: int, val_ratio: float = 0.2, device: Optional[torch.device] = None,
                 n_total: int = 0):
        self.ds_dir_path = ds_dir_path
        self.emb_chunk_size, self.fixed_size = parse_out_subdir(ds_dir_path.name)
        self.docs_batch_size = docs_batch_size
        self.max_chunks_per_doc = max_chunks_per_doc
        self.pad_tok = pad_tok
        self.qbeg_tok = qbeg_tok
        self.qend_tok = qend_tok
        self.df = read_ds_files(ds_dir_path, n_total)
        self.df.set_index(['docid', 'offset'], inplace=True)
        df_doc = self.df.groupby(level=['docid'])
        df_doc = df_doc.agg({'chid': 'count', 'title_tok_num': 'sum', 'body_tok_num': 'sum', 'tok_num': 'sum'})
        df_doc.rename({'chid': 'chunks'}, axis=1, inplace=True)
        self.df_doc = df_doc
        self.val_ratio = val_ratio
        self.docids = self.df_doc.index.to_numpy().copy()
        self.n_docs = len(self.docids)
        self.n_docs_val = int(self.n_docs * self.val_ratio)
        self.n_docs_train = self.n_docs - self.n_docs_val
        self.docids_train = self.docids[:self.n_docs_train].copy()
        self.docids_val = self.docids[self.n_docs_train:].copy()
        self._tokens_cache = {}
        self.device = device
        self.n_total = n_total
        # print(self.df)

    def _prune_cache(self):
        # Relying on dict's property keep keys/values sorted in order of addition
        if len(self._tokens_cache) > self._max_cache_size:
            keys = list(self._tokens_cache.keys())
            cache = self._tokens_cache
            self._tokens_cache = {k: cache[k] for k in keys[-self._max_cache_size:]}

    def _load_tokens(self, doc_id_min: int, doc_id_max: int) -> np.ndarray:
        doc_ids = doc_id_min, doc_id_max
        tokens = self._tokens_cache.get(doc_ids)
        if tokens is None:
            _, tokens_fname, chunk_sizes_fname = gen_ds_fnames(doc_id_min, doc_id_max)
            tokens_fpath, chunk_sizes_fpath = self.ds_dir_path / tokens_fname, self.ds_dir_path / chunk_sizes_fname
            tokens = np.fromfile(tokens_fpath, dtype=np.int32)
            if self.fixed_size:
                tokens = tokens.reshape((-1, self.emb_chunk_size))
            else:
                assert chunk_sizes_fpath.exists(), f'Chunk size is not fixed. File {chunk_sizes_fpath} is not found.'
                chunk_sizes = np.fromfile(chunk_sizes_fpath, dtype=np.int32)
                n_chunks = len(chunk_sizes)
                tokens_list = [None] * n_chunks
                offset = 0
                for i_chunk in range(n_chunks):
                    chunk_size = chunk_sizes[i_chunk]
                    tokens_list[i_chunk] = tokens[offset:offset + chunk_size]
                    offset += chunk_size
                tokens = tokens_list
            self._tokens_cache[doc_ids] = tokens
            self._prune_cache()
            assert doc_ids in self._tokens_cache and len(self._tokens_cache) <= self._max_cache_size
        return tokens

    def _extract_content_tokens(self, df_ch: pd.DataFrame, chunks: list[np.ndarray]) -> list[int]:
        res = []
        for i in range(len(df_ch)):
            ch_row = df_ch.iloc[i]
            ch_tokens = chunks[i]
            title_beg_ind, title_end_ind = ch_row['title_beg_ind'], ch_row['title_end_ind']
            body_beg_ind, body_end_ind = ch_row['body_beg_ind'], ch_row['body_end_ind']
            # print(i, title_beg_ind, title_end_ind, body_beg_ind, body_end_ind)
            # print(len(ch_tokens), ch_tokens[:20])
            if title_beg_ind >= 0:
                assert 0 < title_beg_ind < title_end_ind
                n = len(res)
                res.extend(ch_tokens[title_beg_ind:title_end_ind])
                # print(f'{n} --> {len(res)}')
            if body_beg_ind >= 0:
                assert 0 < body_beg_ind < body_end_ind
                n = len(res)
                res.extend(ch_tokens[body_beg_ind:body_end_ind])
                # print(f'{n} --> {len(res)}')
        # print(f'res: {len(res)}')
        return res

    def get_batch(self, ind: int, train: bool) -> DocsBatch:
        docids = self.docids_train if train else self.docids_val
        docids = docids[ind * self.docs_batch_size:(ind + 1) * self.docs_batch_size]
        df_doc = self.df_doc.loc[docids]
        docs_chunks = {}
        target_tokens = []
        target_docid = np.random.choice(docids)
        for docid in docids:
            n_chunks = df_doc.loc[docid]['chunks']
            df = self.df.loc[docid]
            # print(df)
            i_chunk = 0
            if n_chunks > self.max_chunks_per_doc:
                i_chunk = np.random.randint(n_chunks - self.max_chunks_per_doc)
            df = df.iloc[i_chunk:i_chunk + self.max_chunks_per_doc]
            doc_id_min, doc_id_max = df['doc_id_min'].iloc[0], df['doc_id_max'].iloc[0]

            tokens = self._load_tokens(doc_id_min, doc_id_max)
            chunks = []
            for _, row in df.iterrows():
                chunk_tokens = tokens[row['doc_id_off']]
                chunks.append(chunk_tokens)

            docs_chunks[docid] = chunks

            if docid == target_docid:
                target_tokens = self._extract_content_tokens(df, chunks)
                target_tokens = [self.qbeg_tok, *target_tokens, self.qend_tok]
        return DocsBatch(
            docs_chunks=docs_chunks, target_doc_id=target_docid, target_tokens=target_tokens,
            pad_tok=self.pad_tok, emb_chunk_size=self.emb_chunk_size, fixed_size=self.fixed_size, device=self.device,
        )

    def shuffle(self, train: bool):
        docids = self.docids_train if train else self.docids_val
        np.random.shuffle(docids)


def load_dsfixed():
    ds_path = Path(os.path.expandvars('$HOME')) / 'data' / 'wiki_20200501_en/ch_100_fixed'
    ds = DsLoader(ds_path, 10, 3, 123, 456, 789)


if __name__ == '__main__':
    load_dsfixed()

