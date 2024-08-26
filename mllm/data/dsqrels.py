import itertools
from enum import Enum
from pathlib import Path
from typing import Generator, Optional, TextIO, Union, TypeVar

import numpy as np
import pandas as pd

from mllm.tokenization.chunk_tokenizer import ChunkTokenizer
from mllm.utils.utils import SplitsType, split_range


class DsQrelsId(Enum):
    Msmarco = 1
    Fever = 2


class QrelsBatch:
    def __init__(self, ds_ids: list[int], query_ids: list[int], doc_ids: list[int], ds_query_ids: list[int], ds_doc_ids: list[int],
                 chunks: list[list[ChunkTokenizer.TokChunk]]):
        pass


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
        l = self.fid.readline().rstrip()
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

    def __init__(self, ch_tkz: ChunkTokenizer, ds_ids: list[DsQrelsId], dfs_qs: list[pd.DataFrame], dfs_qrels: list[pd.DataFrame],
                 dfs_off: list[pd.DataFrame], docs_files: dict[DsQrelsId, DocsFile]):
        assert len(ds_ids) == len(dfs_qs) == len(dfs_qrels) == len(dfs_off) == len(docs_files), \
            f'len(ds_ids) = {len(ds_ids)}. len(dfs_qs) = {len(dfs_qs)}. len(dfs_qrels) = {len(dfs_qrels)}. len(dfs_off) = {len(dfs_off)}. len(docs_fids) = {len(docs_files)}'
        self.ch_tkz = ch_tkz
        self.ds_ids = ds_ids
        ds_ids_int = [ds_id.value for ds_id in ds_ids]
        self.df_qs, self.df_qrels, self.df_off = join_qrels_datasets(ds_ids_int, dfs_qs, dfs_qrels, dfs_off)
        self.df_qs.set_index('dsqid', inplace=True)
        self.df_qrels.set_index('dsqid', inplace=True)
        self.df_off.set_index('dsdid', inplace=True)
        self.docs_files = docs_files

    def get_view(self, batch_size: Optional[int] = None) -> DsQrelsView:
        ids = self.df_qs.index.values
        return DsQrelsView(self, ids, batch_size)

    def _get_doc_title_text(self, ds_id: int, offset: int) -> tuple[str, str]:
        ds_id_ = DsQrelsId(ds_id)
        docs_file = self.docs_files[ds_id_]
        l = docs_file.get_line(offset)
        _, _, title, text = l.rstrip().split('\t')
        return title, text

    def get_batch(self, dsqids: np.ndarray) -> QrelsBatch:
        ds_ids, query_ids, doc_ids, ds_query_ids, ds_doc_ids, chunks = [], [], [], [], [], []
        for dsqid in dsqids:
            q_row = self.df_qs.loc[dsqid]
            qr_rows = self.df_qrels.loc[dsqid]
            qr_row = qr_rows.sample(n=1)
            off_row = self.df_off.loc[qr_row['dsdid']]
            title, text = self._get_doc_title_text(q_row['dsid'], off_row['offset'])
            tok_chunks = self.ch_tkz.process_doc(qr_row['dsdid'], {'title': title, 'text': text})
            ds_ids.append(q_row['dsid'])
            query_ids.append(q_row['qid'])
            doc_ids.append(qr_row['did'])
            ds_query_ids.append(q_row['dsqid'])
            ds_doc_ids.append(qr_row['dsdid'])
            chunks.append(tok_chunks)
        return QrelsBatch(ds_ids=ds_ids, query_ids=query_ids, doc_ids=doc_ids, ds_query_ids=ds_query_ids, ds_doc_ids=ds_doc_ids, chunks=chunks)

    @staticmethod
    def join(dss: list['DsQrels']) -> 'DsQrels':
        ds_ids = list(itertools.chain.from_iterable(ds.ds_ids for ds in dss))
        assert len(ds_ids) == len(set(ds_ids)), f'{ds_ids}'

    def close(self):
        for docs_file in self.docs_files.values():
            docs_file.close()

