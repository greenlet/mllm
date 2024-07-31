import os.path
import shutil
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import torch
from tqdm import trange

from mllm.data.common import DocsBatch
from mllm.tokenization.chunk_tokenizer import parse_out_subdir, gen_ds_fnames


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


def extract_content_tokens(df_chunks: pd.DataFrame, chunks: list[np.ndarray]) -> list[list[int]]:
    res = []
    for i in range(len(df_chunks)):
        ch_row = df_chunks.iloc[i]
        ch_tokens = chunks[i]
        title_beg_ind, title_end_ind = ch_row['title_beg_ind'], ch_row['title_end_ind']
        body_beg_ind, body_end_ind = ch_row['body_beg_ind'], ch_row['body_end_ind']
        # print(i, title_beg_ind, title_end_ind, body_beg_ind, body_end_ind)
        # print(len(ch_tokens), ch_tokens[:20])
        toks = []
        if title_beg_ind >= 0:
            assert 0 < title_beg_ind < title_end_ind
            toks.extend(list(ch_tokens[title_beg_ind:title_end_ind]))
        if body_beg_ind >= 0:
            assert 0 < body_beg_ind < body_end_ind
            toks.extend(list(ch_tokens[body_beg_ind:body_end_ind]))
        res.append(toks)
    # print(f'res: {len(res)}')
    return res


class DsLoader:
    ds_path: Path
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

    def __init__(self, ds_path: Path, docs_batch_size: int, max_chunks_per_doc: int,
                 pad_tok: int, qbeg_tok: int, qend_tok: int, val_ratio: float = 0.2, device: Optional[torch.device] = None,
                 n_total: int = 0):
        self.ds_path = ds_path
        self.emb_chunk_size, self.fixed_size = parse_out_subdir(ds_path.name)
        self.docs_batch_size = docs_batch_size
        self.max_chunks_per_doc = max_chunks_per_doc
        self.pad_tok = pad_tok
        self.qbeg_tok = qbeg_tok
        self.qend_tok = qend_tok
        self.df = read_ds_files(ds_path, n_total)
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
            tokens_fpath, chunk_sizes_fpath = self.ds_path / tokens_fname, self.ds_path / chunk_sizes_fname
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

    def get_batch(self, ind: int, train: bool, target_augmenter: Optional[Callable] = None) -> DocsBatch:
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
                target_tokens = extract_content_tokens(df, chunks)
                if target_augmenter is not None:
                    target_tokens = target_augmenter(target_tokens)

        return DocsBatch(
            docs_chunks=docs_chunks, target_doc_id=target_docid, target_tokens=target_tokens,
            pad_tok=self.pad_tok, qbeg_tok=self.qbeg_tok, qend_tok=self.qend_tok,
            emb_chunk_size=self.emb_chunk_size, fixed_size=self.fixed_size, device=self.device,
        )

    def shuffle(self, train: bool):
        docids = self.docids_train if train else self.docids_val
        np.random.shuffle(docids)


def load_dsfixed():
    ds_path = Path(os.path.expandvars('$HOME')) / 'data' / 'wiki_20200501_en/ch_100_fixed'
    ds = DsLoader(ds_path, 10, 3, 123, 456, 789)


if __name__ == '__main__':
    load_dsfixed()

