from pathlib import Path
from typing import Optional, TextIO, Type, BinaryIO, Union

import numpy as np
import pandas as pd
import torch

from mllm.data.common import DsView
from mllm.utils.utils import read_tsv



class QrelsEmbsBatch:
    # doc_emb_id: int (index), ds_id: int, ds_doc_id: int
    df_docs_ids: pd.DataFrame
    # docs_embs: [n_batch, chunk_size, emb_size]
    docs_embs: np.ndarray
    # qid: int, did: int, dsqid: int (generated), dsdid: int (generated)
    batch_size: int
    chunk_size: int
    emb_size: int
    device: Optional[torch.device] = None
    docs_embs_tf: Optional[torch.Tensor] = None
    df_qrels: Optional[pd.DataFrame]
    # query_emb_id: int (index), ds_id: int, ds_query_id: int
    df_qs_ids: Optional[pd.DataFrame]
    qs_embs: Optional[list[np.ndarray]]
    qs_embs_tf: Optional[torch.Tensor] = None
    masks_embs_tf: Optional[torch.Tensor] = None

    def __init__(
            self, df_docs_ids: pd.DataFrame, docs_embs: list[np.ndarray], chunk_size: int, emb_size: int, device: Optional[torch.device] = None,
            df_qrels: Optional[pd.DataFrame] = None, df_qs_ids: Optional[pd.DataFrame] = None, qs_embs: Optional[list[np.ndarray]] = None
    ):
        self.df_docs_ids = df_docs_ids
        # [n_batch * chunk_size, emb_size]
        docs_embs = np.stack(self.docs_embs, axis=0)
        # [n_batch, chunk_size, emb_size]
        docs_embs = docs_embs.reshape((-1, self.chunk_size, self.emb_size))
        self.docs_embs = docs_embs
        self.batch_size = len(self.docs_embs)
        self.chunk_size = chunk_size
        self.emb_size = emb_size
        self.device = device
        self.df_qrels = df_qrels
        self.df_qs_ids = df_qs_ids
        self.qs_embs = qs_embs
        self._calc()

    def _calc(self):
        # [n_batch * chunk_size]
        ds_docs_ids = self.df_docs_ids['ds_doc_id'].to_numpy()
        # [n_batch, chunk_size]
        ds_docs_ids = ds_docs_ids.reshape((self.batch_size, self.chunk_size))
        df_qrels = self.df_qrels.set_index('dsdid')
        batch_did_to_qid = {}
        for i_batch, batch_dsdids in enumerate(ds_docs_ids):
            did_to_qid = {}
            batch_did_to_qid[i_batch] = did_to_qid
            dsdids = np.unique(batch_dsdids)
            dsqids = df_qrels['ds_query_id']
            for dsdid in dsdids:
                dsdidqids = list(dsqids.loc[dsdid])
                did_to_qid[dsdid] = dsdidqids


    def get_docs_tensor(self, with_qs_ids: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, dict[int, int]]]:
        # [n_batch * chunk_size, emb_size]
        docs_embs = np.stack(self.docs_embs, axis=0)
        # [n_batch, chunk_size, emb_size]
        docs_embs = docs_embs.reshape((-1, self.chunk_size, self.emb_size))
        n_batch = len(docs_embs)



        res = torch.from_numpy(docs_embs)
        if self.device is not None:
            res = res.to(self.device)
        return res

    def _get_qs_tensors(self) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        n_qs = len(self.df_qs_ids)
        dsqid_last, i_start = None, None
        qs_off_len = []
        df_qrels = self.df_qrels.set_index('dsqid')
        for i in range(n_qs):
            qrow = self.df_qs_ids.iloc[i]
            dsqid = qrow['ds_query_id']
            if dsqid_last is None:
                dsqid_last, i_start = dsqid, i
            if dsqid_last != dsqid:
                qs_off_len.append((dsqid_last, i_start, i - i_start))
                dsqid_last, i_start = dsqid, i

    def get_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        if self.docs_embs_tf is None:
            self.docs_embs_tf = self.get_docs_tensor(self.docs_embs)


class DsQrelsEmbsView(DsView['DsQrelsEmbs', QrelsEmbsBatch]):
    pass


class BinVecsFile:
    fpath: Path
    fid: BinaryIO
    vec_size: int
    dtype: np.dtype[int]
    bytes_size: int
    opened: bool

    def __init__(self, fpath: Path, vec_size: int, dtype: np.dtype[int]):
        self.fpath = fpath
        self.fid = open(self.fpath, 'rb')
        self.opened = True
        self.vec_size = vec_size
        self.dtype = np.dtype(dtype)
        self.bytes_size = self.vec_size * self.dtype.itemsize

    def get_vec(self, offset: int) -> np.ndarray:
        assert self.opened
        self.fid.seek(offset)
        buf = self.fid.read(self.bytes_size)
        vec = np.frombuffer(buf, self.dtype)
        return vec

    def close(self):
        if self.opened:
            self.fid.close()
            self.opened = False


class DsQrelsEmbs:
    ds_dir_path: Path
    chunk_size: int
    emb_size: int
    emb_dtype: np.dtype[int]
    emb_bytes_size: int
    # doc_emb_id: int (index), ds_id: int, ds_doc_id: int
    df_docs_ids: pd.DataFrame
    # query_emb_id: int (index), ds_id: int, ds_query_id: int
    df_qs_ids: pd.DataFrame
    # qid: int, did: int, dsqid: int (generated), dsdid: int (generated)
    df_qrels: pd.DataFrame
    docs_embs_file: BinVecsFile
    qs_file: BinVecsFile
    device: Optional[torch.device] = None

    def __init__(self, ds_dir_path: Path, chunk_size: int, emb_size: int, emb_dtype: np.dtype[int], device: Optional[torch.device] = None):
        self.ds_dir_path = ds_dir_path
        self.chunk_size = chunk_size
        self.emb_size = emb_size
        self.emb_dtype = emb_dtype
        self.emb_bytes_size = self.emb_size * np.dtype(self.emb_dtype).itemsize
        self.device = device
        docs_ids_fpath = self.ds_dir_path / 'docs_ids.tsv'
        docs_embs_fpath = self.ds_dir_path / 'docs_embs.npy'
        qs_ids_fpath = self.ds_dir_path / 'qs_ids.tsv'
        qs_embs_fpath = self.ds_dir_path / 'qs_embs.npy'
        qrels_fpath = self.ds_dir_path / 'qrels.tsv'
        self.df_docs_ids = read_tsv(docs_ids_fpath)
        self.df_docs_ids.set_index('doc_emb_id', inplace=True)
        self.df_qs_ids = read_tsv(qs_ids_fpath)
        self.df_qs_ids.set_index('ds_query_id', inplace=True)
        self.df_qrels = read_tsv(qrels_fpath)
        self.df_qrels.set_index('dsdid', inplace=True)
        self.docs_embs_file = BinVecsFile(fpath=docs_embs_fpath, vec_size=self.emb_size, dtype=self.emb_dtype)
        self.qs_embs_file = BinVecsFile(fpath=qs_embs_fpath, vec_size=self.emb_size, dtype=self.emb_dtype)

    # doc_emb_id: int (index), ds_id: int, ds_doc_id: int
    def _get_qs_embs(self, df_docs_ids: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame, list[np.ndarray]]:
        ds_doc_ids = df_docs_ids['ds_doc_id']
        ds_doc_ids = np.unique(ds_doc_ids)
        df_qrels = self.df_qrels.loc[ds_doc_ids]
        ds_qs_ids = np.unique(df_qrels['dsqid'])
        df_qs_ids = self.df_qs_ids.loc[ds_qs_ids]
        qs_embs: list[np.ndarray] = []
        for _, row in df_qs_ids.iterrows():
            off = row['query_emb_id'] * self.emb_bytes_size
            query_emb = self.qs_embs_file.get_vec(off)
            qs_embs.append(query_emb)
        return df_qrels, self.df_qs_ids, qs_embs

    def get_embs_batch(self, doc_emb_ids: np.ndarray, with_queries: bool = True) -> QrelsEmbsBatch:
        df_docs_ids = self.df_docs_ids.loc[doc_emb_ids]
        offsets = doc_emb_ids * self.emb_bytes_size
        docs_embs: list[np.ndarray] = []
        for off in offsets:
            doc_emb = self.docs_embs_file.get_vec(off)
            docs_embs.append(doc_emb)
        df_qrels, df_qs_ids, qs_embs = None, None, None
        if with_queries:
            df_qrels, df_qs_ids, qs_embs = self._get_qs_embs(df_docs_ids)
        batch = QrelsEmbsBatch(
            df_docs_ids=df_docs_ids, docs_embs=docs_embs, df_qrels=df_qrels, df_qs_ids=df_qs_ids, qs_embs=qs_embs,
            chunk_size=self.chunk_size, emb_size=self.emb_size, device=self.device
        )
        return batch

    def get_embs_view(self, batch_size: Optional[int] = None) -> DsQrelsEmbsView:
        ids = self.df_docs_ids.index.values
        return DsQrelsEmbsView(self, ids, self.get_embs_batch, batch_size * self.chunk_size)

    def close(self):
        self.docs_embs_file.close()
        self.qs_embs_file.close()

    def __str__(self) -> str:
        return f'{self.ds_dir_path.name}. Docs embeddings: {len(self.df_docs_ids)}. Queries embeddings: {len(self.df_qs_ids)}'

    def __repr__(self) -> str:
        return str(self)


