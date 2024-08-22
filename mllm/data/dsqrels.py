import itertools
from enum import Enum
from pathlib import Path
from typing import Generator, Optional, TextIO, Union, TypeVar

import numpy as np
import pandas as pd


class DsQrelsId(Enum):
    Msmarco = 1
    Fever = 2


class QrelsBatch:
    pass


class DsQrelsView:
    ds: 'DsQrels'
    ids: np.ndarray
    batch_size: Optional[int] = None

    def __init__(self, ds: 'DsQrels', ids: np.ndarray, batch_size: Optional[int] = None):
        self.ds = ds
        self.ids = ids
        self.batch_size = batch_size

    def split(self, first_ratio: float) -> tuple['DsQrelsView', 'DsQrelsView']:
        pass

    def get_batch_it(self, batch_size: Optional[None]) -> Generator[QrelsBatch, None, None]:
        pass


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
        df_qs['dsqid'] = np.arange(qid_off, qid_off + len(df_qs))
        df_qid_to_dsqid = df_qs[['dsqid', 'qid']].set_index('qid')['dsqid']
        df_off['dsid'] = ds_id
        df_off['dsdid'] = np.arange(did_off, did_off + len(df_off))
        df_did_to_dsdid = df_off[['dsdid', 'did']].set_index('did')['dsdid']
        df_qrels['dsid'] = ds_id
        df_qrels['dsqid'] = df_qrels['qid'].map(df_qid_to_dsqid)
        df_qrels['dsdid'] = df_qrels['did'].map(df_did_to_dsdid)
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
    ds_ids: list[DsQrelsId]
    # qid: int, query: str, dsid: int (added), dsqid: int (generated)
    df_qs: pd.DataFrame
    # qid: int, did: int, dsqid: int (generated), dsdid: int (generated)
    df_qrels: pd.DataFrame
    # did: int, offset: int, dsdid: int (generated), dsid: int (added)
    df_off: pd.DataFrame
    docs_files: dict[DsQrelsId, DocsFile]

    def __init__(self, ds_ids: list[DsQrelsId], dfs_qs: list[pd.DataFrame], dfs_qrels: list[pd.DataFrame],
                 dfs_off: list[pd.DataFrame], docs_files: dict[DsQrelsId, DocsFile]):
        assert len(ds_ids) == len(dfs_qs) == len(dfs_qrels) == len(dfs_off) == len(docs_files), \
            f'len(ds_ids) = {len(ds_ids)}. len(dfs_qs) = {len(dfs_qs)}. len(dfs_qrels) = {len(dfs_qrels)}. len(dfs_off) = {len(dfs_off)}. len(docs_fids) = {len(docs_files)}'
        self.ds_ids = ds_ids
        if len(dfs_qs) == 1:
            self.df_qs = dfs_qs[0]
            self.df_qrels = dfs_qrels[0]
            self.df_off = dfs_off[0]
        else:
            ds_ids_int = [ds_id.value for ds_id in ds_ids]
            self.df_qs, self.df_qrels, self.df_off = join_qrels_datasets(ds_ids_int, dfs_qs, dfs_qrels, dfs_off)
        self.docs_files = docs_files

    def get_view(self) -> DsQrelsView:
        pass

    @staticmethod
    def join(dss: list['DsQrels']) -> 'DsQrels':
        ds_ids = list(itertools.chain.from_iterable(ds.ds_ids for ds in dss))
        assert len(ds_ids) == len(set(ds_ids)), f'{ds_ids}'

    def close(self):
        for docs_file in self.docs_files.values():
            docs_file.close()