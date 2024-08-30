import itertools
from enum import Enum
from pathlib import Path
from typing import Generator, Optional, TextIO, Union, TypeVar

import numpy as np
import pandas as pd
import torch

from mllm.tokenization.chunk_tokenizer import ChunkTokenizer, split_doc_embs
from mllm.utils.utils import SplitsType, split_range


class DsQrelsId(Enum):
    Msmarco = 1
    Fever = 2


class QrelsBatch:
    df_qs: pd.DataFrame
    docs_chunks: list[list[np.ndarray]]
    qs_chunks: list[list[list[int]]]
    pad_tok: int
    emb_chunk_size: int
    docs_chunks_padded: np.ndarray
    qs_chunks_padded: np.ndarray
    docs_off_len: list[tuple[int, int]]
    qs_off_len: list[tuple[int, int]]
    docs_chunks_padded_tf: Optional[torch.Tensor] = None
    qs_chunks_padded_tf: Optional[torch.Tensor] = None
    device: Optional[torch.device] = None

    def __init__(self, df_qs: pd.DataFrame, docs_chunks: list[list[np.ndarray]], qs_chunks: list[list[list[int]]], pad_tok: int, emb_chunk_size: int,
                 device: Optional[torch.device] = None):
        self.df_qs = df_qs
        self.docs_chunks = docs_chunks
        self.qs_chunks = qs_chunks
        self.pad_tok = pad_tok
        self.emb_chunk_size = emb_chunk_size
        self.device = device
        self.calc_np()

    def calc_np(self):
        assert len(self.docs_chunks) == len(self.qs_chunks), f'# of docs ({len(self.docs_chunks)}) != # of queries ({len(self.qs_chunks)})'

        max_chunks_sz = 0
        docs_off_len, docs_off = [], 0
        for chunk in self.docs_chunks:
            max_chunks_sz = max(max_chunks_sz, max(len(tokens) for tokens in chunk))
            n_chunk = len(chunk)
            docs_off_len.append((docs_off, n_chunk))
            docs_off += n_chunk

        qs_off_len, qs_off = [], 0
        for chunk in self.qs_chunks:
            max_chunks_sz = max(max_chunks_sz, max(len(tokens) for tokens in chunk))
            n_chunk = len(chunk)
            qs_off_len.append((qs_off, n_chunk))
            qs_off += n_chunk

        docs_chunks_padded = np.full((docs_off, max_chunks_sz), self.pad_tok, dtype=np.int32)
        for i_doc, doc_chunk in enumerate(self.docs_chunks):
            for i_tok, tokens in enumerate(doc_chunk):
                off = docs_off_len[i_doc][0]
                docs_chunks_padded[off + i_tok, :len(tokens)] = tokens

        qs_chunks_padded = np.full((qs_off, max_chunks_sz), self.pad_tok, dtype=np.int32)
        for i_query, query_chunk in enumerate(self.qs_chunks):
            for i_tok, tokens in enumerate(query_chunk):
                off = qs_off_len[i_query][0]
                qs_chunks_padded[off + i_tok, :len(tokens)] = tokens

        self.docs_chunks_padded = docs_chunks_padded
        self.qs_chunks_padded = qs_chunks_padded
        self.docs_off_len = docs_off_len
        self.qs_off_len = qs_off_len

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        res = torch.from_numpy(arr)
        if self.device is not None:
            res = res.to(self.device)
        return res

    def gen_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.docs_chunks_padded_tf is None:
            self.docs_chunks_padded_tf, self.qs_chunks_padded_tf = \
                map(self._to_tensor, (self.docs_chunks_padded, self.qs_chunks_padded))
        return self.docs_chunks_padded_tf, self.qs_chunks_padded_tf


class DsQrelsView:
    ds: 'DsQrels'
    ids: np.ndarray
    batch_size: Optional[int] = None

    def __init__(self, ds: 'DsQrels', ids: np.ndarray, batch_size: Optional[int] = None):
        self.ds = ds
        self.ids = ids.copy()
        self.batch_size = batch_size

    def split(self, splits: SplitsType) -> tuple['DsQrelsView', ...]:
        intervals = split_range(len(self.ids), splits)
        res = []
        for i in range(1, len(intervals)):
            ids = self.ids[intervals[i - 1]:intervals[i]]
            ov = DsQrelsView(
                ds=self.ds, ids=ids, batch_size=self.batch_size,
            )
            res.append(ov)
        return tuple(res)

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def get_batch_iterator(self, n_batches: Optional[int] = None, batch_size: Optional[int] = None,
                           drop_last: bool = False, shuffle_between_loops: bool = True)\
            -> Generator[QrelsBatch, None, None]:
        batch_size = batch_size or self.batch_size
        n = len(self.ids)
        n_batches_total = n // batch_size + min(n % batch_size, 1)

        info = f'n = {n}. batch_size = {batch_size}. n_batches = {n_batches}. n_batches_total = {n_batches_total}'
        assert n_batches_total > 0, info
        assert n_batches is None or n_batches > 0, info

        looped = False
        if n_batches is None:
            n_batches = n_batches_total
        if n_batches > n_batches_total:
            looped = True

        for i_batch in range(n_batches):
            i = i_batch * batch_size
            if i >= n:
                if shuffle_between_loops:
                    np.random.shuffle(self.ids)
                    i = 0
                else:
                    i %= n
            batch_size_cur = min(batch_size, n - i)
            inds = range(i, i + batch_size_cur)
            if batch_size_cur < batch_size:
                if not looped:
                    if drop_last:
                        return
                else:
                    rest = batch_size - batch_size_cur
                    inds = list(range(i, n)) + list(range(rest))
            ids = self.ids[inds]
            batch = self.ds.get_batch(ids)
            yield batch

    def __len__(self) -> int:
        return len(self.ids)


class DocsFile:
    fpath: Path
    fid: TextIO
    opened: bool

    def __init__(self, fpath: Path):
        self.fpath = fpath
        self.fid = open(self.fpath, 'r', encoding='utf-8')
        self.opened = True

    def get_line(self, offset: int) -> str:
        assert self.opened
        self.fid.seek(offset)
        l = self.fid.readline().rstrip('\n')
        return l

    def close(self):
        if self.opened:
            self.fid.close()
            self.opened = False


def join_qrels_datasets(
        ds_ids: list[int], dfs_qs: list[pd.DataFrame], dfs_qrels: list[pd.DataFrame],
        dfs_off: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dfs_qs_new, dfs_qrels_new, dfs_off_new = [], [], []
    qid_off, did_off = 0, 0
    for ds_id, df_qs, df_qrels, df_off in zip(ds_ids, dfs_qs, dfs_qrels, dfs_off):
        df_qs, df_qrels, df_off = df_qs.copy(), df_qrels.copy(), df_off.copy()
        df_qs['dsid'] = ds_id
        df_qs['dsqid'] = np.arange(qid_off, qid_off + len(df_qs), dtype=int)
        df_qid_to_dsqid = df_qs[['dsqid', 'qid']].set_index('qid')['dsqid']
        df_off['dsid'] = ds_id
        df_off['dsdid'] = np.arange(did_off, did_off + len(df_off), dtype=int)
        df_did_to_dsdid = df_off[['dsdid', 'did']].set_index('did')['dsdid']
        df_qrels['dsid'] = ds_id
        df_qrels['dsqid'] = df_qrels['qid'].map(df_qid_to_dsqid).astype(int)
        df_qrels['dsdid'] = df_qrels['did'].map(df_did_to_dsdid).astype(int)
        dfs_qs_new.append(df_qs)
        dfs_qrels_new.append(df_qrels)
        dfs_off_new.append(df_off)
        qid_off += len(df_qs)
        did_off += len(df_off)
    df_qs_res = pd.concat(dfs_qs_new, axis=0)
    df_qrels_res = pd.concat(dfs_qrels_new, axis=0)
    df_off_res = pd.concat(dfs_off_new, axis=0)
    return df_qs_res, df_qrels_res, df_off_res


class DsQrels:
    ch_tkz: ChunkTokenizer
    ds_ids: list[DsQrelsId]
    # qid: int, query: str, dsid: int (added), dsqid: int (generated)
    df_qs: pd.DataFrame
    # qid: int, did: int, dsqid: int (generated), dsdid: int (generated)
    df_qrels: pd.DataFrame
    # did: int, offset: int, dsdid: int (generated), dsid: int (added)
    df_off: pd.DataFrame
    docs_files: dict[DsQrelsId, DocsFile]
    max_chunks_per_doc: int
    emb_chunk_size: int
    device: Optional[torch.device] = None

    def __init__(self, ch_tkz: ChunkTokenizer, ds_ids: list[DsQrelsId], dfs_qs: list[pd.DataFrame], dfs_qrels: list[pd.DataFrame],
                 dfs_off: list[pd.DataFrame], docs_files: dict[DsQrelsId, DocsFile],
                 max_chunks_per_doc: int, emb_chunk_size: int, device: Optional[torch.device] = None):
        assert len(ds_ids) == len(dfs_qs) == len(dfs_qrels) == len(dfs_off) == len(docs_files), \
            f'len(ds_ids) = {len(ds_ids)}. len(dfs_qs) = {len(dfs_qs)}. len(dfs_qrels) = {len(dfs_qrels)}. len(dfs_off) = {len(dfs_off)}. len(docs_fids) = {len(docs_files)}'
        self.ch_tkz = ch_tkz
        self.ds_ids = ds_ids
        ds_ids_int = [ds_id.value for ds_id in ds_ids]
        self.df_qs, self.df_qrels, self.df_off = join_qrels_datasets(ds_ids_int, dfs_qs, dfs_qrels, dfs_off)
        self.df_qs.set_index('dsqid', inplace=True, drop=False)
        self.df_qrels.set_index('dsqid', inplace=True, drop=False)
        self.df_off.set_index('dsdid', inplace=True, drop=False)
        self.docs_files = docs_files
        self.max_chunks_per_doc = max_chunks_per_doc
        self.emb_chunk_size = emb_chunk_size
        self.device = device

    def get_view(self, batch_size: Optional[int] = None) -> DsQrelsView:
        ids = self.df_qs.index.values
        return DsQrelsView(self, ids, batch_size)

    def _get_doc_title_text(self, ds_id: int, offset: int) -> tuple[str, str]:
        ds_id_ = DsQrelsId(ds_id)
        docs_file = self.docs_files[ds_id_]
        l = docs_file.get_line(offset)
        _, url, title, text = l.split('\t')
        text = text if text else ' '.join(url.split('/'))
        return title, text

    def _tokenize_query(self, query: str) -> list[list[int]]:
        tokens = self.ch_tkz.tokenizer(query)['input_ids']
        # tokens = [self.qbeg_tok, *tokens, self.qend_tok]
        off = split_doc_embs(len(tokens), self.emb_chunk_size, fixed_size=True)
        res = []
        for i in range(len(off) - 1):
            res.append(tokens[off[i]:off[i + 1]])
        return res

    def get_batch(self, dsqids: np.ndarray) -> QrelsBatch:
        ds_ids, query_ids, doc_ids, ds_query_ids, ds_doc_ids, chunks = [], [], [], [], [], []
        df_qs = self.df_qs.loc[dsqids].copy()
        df_qs['did'] = 0
        df_qs['dsdid'] = 0
        df_qs['title'] = ''
        df_qs['text'] = ''
        docid_sequential = 0
        qs_chunks, docs_chunks = [], []
        for dsqid in dsqids:
            q_row = df_qs.loc[dsqid]
            qr_rows = self.df_qrels.loc[dsqid]
            if type(qr_rows) == pd.Series:
                qr_row = qr_rows
            else:
                qr_row = qr_rows.sample(n=1).iloc[0]
            off_row = self.df_off.loc[qr_row['dsdid']]
            title, text = self._get_doc_title_text(q_row['dsid'], off_row['offset'])
            df_qs.loc[dsqid, 'did'] = off_row['did']
            df_qs.loc[dsqid, 'dsdid'] = off_row['dsdid']
            df_qs.loc[dsqid, 'title'] = title
            df_qs.loc[dsqid, 'text'] = text
            docid = docid_sequential
            doc_chunks = self.ch_tkz.process_doc(docid, {'title': title, 'text': text})
            if len(doc_chunks) > self.max_chunks_per_doc:
                i = np.random.randint(len(doc_chunks) - self.max_chunks_per_doc + 1)
                doc_chunks = doc_chunks[i:i + self.max_chunks_per_doc]
            query = q_row['query']
            query_chunks = self._tokenize_query(query)
            docs_chunks.append([ch.tokens for ch in doc_chunks])
            qs_chunks.append(query_chunks)
            docid_sequential += 1

        return QrelsBatch(
            df_qs=df_qs, docs_chunks=docs_chunks, qs_chunks=qs_chunks, pad_tok=self.ch_tkz.pad_tok, emb_chunk_size=self.emb_chunk_size,
            device=self.device,
        )

    @staticmethod
    def join(dss: list['DsQrels']) -> 'DsQrels':
        ds_ids = list(itertools.chain.from_iterable(ds.ds_ids for ds in dss))
        assert len(ds_ids) == len(set(ds_ids)), f'{ds_ids}'
        ch_tkz: Optional[ChunkTokenizer] = None
        ds_ids: list[DsQrelsId] = []
        dfs_qs: list[pd.DataFrame] = []
        dfs_qrels: list[pd.DataFrame] = []
        dfs_off: list[pd.DataFrame] = []
        docs_files: dict[DsQrelsId, DocsFile] = {}
        max_chunks_per_doc: Optional[int] = None
        emb_chunk_size: Optional[int] = None
        device: Optional[torch.device] = None
        first = True
        for ds in dss:
            if first:
                ch_tkz, max_chunks_per_doc, emb_chunk_size, device = ds.ch_tkz, ds.max_chunks_per_doc, ds.emb_chunk_size, ds.device
                first = False
            else:
                assert ds.ch_tkz == ch_tkz
                assert ds.max_chunks_per_doc == max_chunks_per_doc
                assert ds.emb_chunk_size == emb_chunk_size
                assert ds.device == device
            ds_ids.extend(ds.ds_ids)
            dfs_qs.append(ds.df_qs)
            dfs_qrels.append(ds.df_qrels)
            dfs_off.append(ds.df_off)
            docs_files = {
                **docs_files,
                **ds.docs_files,
            }
        return DsQrels(
            ch_tkz=ch_tkz, ds_ids=ds_ids, dfs_qs=dfs_qs, dfs_qrels=dfs_qrels, dfs_off=dfs_off, docs_files=docs_files,
            max_chunks_per_doc=max_chunks_per_doc, emb_chunk_size=emb_chunk_size, device=device,
        )

    def close(self):
        for docs_file in self.docs_files.values():
            docs_file.close()

    def __str__(self) -> str:
        ids_str = ','.join(ds_id.name for ds_id in self.ds_ids)
        return f'{ids_str}. Queries: {len(self.df_qs)}. Docs: {len(self.df_off)}. QueryDocRels: {len(self.df_qrels)}'

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self.df_qs)

