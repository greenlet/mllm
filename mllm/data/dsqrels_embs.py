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
    # [n_batch * chunk_size, 2]
    docs_embs_ids: np.ndarray
    batch_size: int
    chunk_size: int
    emb_size: int
    device: Optional[torch.device] = None
    docs_embs_t: Optional[torch.Tensor] = None

    def __init__(
            self, df_docs_ids: pd.DataFrame, docs_embs: list[np.ndarray], chunk_size: int, emb_size: int,
            device: Optional[torch.device] = None,
    ):
        self.df_docs_ids = df_docs_ids

        # [n_batch * chunk_size, emb_size]
        docs_embs = np.stack(docs_embs, axis=0)

        # last batch might contain number of embeddings which is not a multiple of chunk_size
        n_docs = len(docs_embs)
        n_docs_chunk_rem = n_docs % chunk_size
        if n_docs_chunk_rem > 0:
            docs_embs = np.pad(docs_embs, ((0, chunk_size - n_docs_chunk_rem), (0, 0)))

        # [n_batch, chunk_size, emb_size]
        docs_embs = docs_embs.reshape((-1, chunk_size, emb_size))
        # [n_batch * chunk_size]
        doc_emb_id = self.df_docs_ids.index.to_numpy()
        if n_docs_chunk_rem > 0:
            doc_emb_id = np.pad(doc_emb_id, (0, chunk_size - n_docs_chunk_rem), constant_values=-1)

        batch_size = len(docs_embs)
        doc_emb_id_1 = np.arange(batch_size)
        doc_emb_id_1 = np.repeat(doc_emb_id_1, chunk_size)
        docs_embs_ids = np.stack([doc_emb_id, doc_emb_id_1], axis=1)
        self.docs_embs = docs_embs
        self.docs_embs_ids = docs_embs_ids
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.emb_size = emb_size
        self.device = device

    def _calc_old(self):
        # # qid: int, did: int, dsqid: int (generated), dsdid: int (generated)
        # df_qrels: Optional[pd.DataFrame] = None
        # # query_emb_id: int (index), ds_id: int, ds_query_id: int
        # df_qs_ids: Optional[pd.DataFrame] = None
        # did_to_qids: Optional[list[dict[int, list[int]]]] = None
        # qids: Optional[list[set[int]]] = None
        # qs_ind_len: Optional[list[tuple[int, int, int]]]

        # [n_batch * chunk_size]
        ds_docs_ids = self.df_docs_ids['ds_doc_id'].to_numpy()
        # [n_batch, chunk_size]
        ds_docs_ids = ds_docs_ids.reshape((self.batch_size, self.chunk_size))
        df_qrels = self.df_qrels.set_index('dsdid')
        did_to_qids, qids = [], []
        for i_batch, ds_docs_ids_b in enumerate(ds_docs_ids):
            did_to_qids_b, qids_b = {}, set()
            ds_docs_ids_b = np.unique(ds_docs_ids_b)
            ds_qs_ids_b = df_qrels['ds_query_id']
            for dsdid in ds_docs_ids_b:
                dsdidqids = ds_qs_ids_b.loc[dsdid]
                dsdidqids = list(dsdidqids) if type(dsdidqids) == pd.Series else [dsdidqids]
                did_to_qids_b[dsdid] = dsdidqids
                qids_b.update(dsdidqids)
            did_to_qids.append(did_to_qids_b)
            qids.append(qids_b)
        self.did_to_qids = did_to_qids
        self.qids = qids

        qs_ind_len = []
        q_ind, q_len = 0, 0
        qid_last = None
        self.df_qs_ids.sort_values(['ds_query_id', 'query_emb_id'], inplace=True)
        for i, (_, q_row) in enumerate(self.df_qs_ids.iterrows()):
            qid = q_row['ds_query_id']
            if qid_last != qid:
                if qid_last is not None:
                    qs_ind_len.append((qid, q_ind, q_len))
                q_ind, q_len = i, 0
                qid_last = qid
            q_len += 1
        qs_ind_len.append((qid_last, q_ind, q_len))
        self.qs_ind_len = qs_ind_len

        # n_qs = len(qs_ind_len)
        # qs_masks = np.zeros((self.batch_size, n_qs), dtype=bool)
        # for i_q, (qid, _, _) in enumerate(qs_ind_len):
        #     for i_b, qids_b in enumerate(qids):
        #         if qid in qids_b:
        #             qs_masks[i_b, i_q] = True
        # self.qs_masks = qs_masks

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        res = torch.from_numpy(arr)
        if self.device is not None:
            res = res.to(self.device)
        return res

    def get_docs_embs_tensor(self) -> torch.Tensor:
        if self.docs_embs_t is None:
            self.docs_embs_t = self._to_tensor(self.docs_embs)
        return self.docs_embs_t


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
        self.df_qs_ids = read_tsv(qs_ids_fpath)
        self.df_qs_ids.set_index('ds_query_id', inplace=True)
        self.df_qrels = read_tsv(qrels_fpath)
        self.df_qrels.set_index('dsdid', inplace=True)
        df_docs_ids = read_tsv(docs_ids_fpath)
        df_docs_ids.set_index('ds_doc_id', inplace=True)
        df_docs_ids = df_docs_ids.loc[self.df_qrels.index.unique().to_numpy()].copy()
        df_docs_ids.reset_index(drop=False, inplace=True)
        df_docs_ids.set_index('doc_emb_id', inplace=True)
        self.df_docs_ids = df_docs_ids
        self.docs_embs_file = BinVecsFile(fpath=docs_embs_fpath, vec_size=self.emb_size, dtype=self.emb_dtype)
        self.qs_embs_file = BinVecsFile(fpath=qs_embs_fpath, vec_size=self.emb_size, dtype=self.emb_dtype)

    # doc_emb_id: int (index), ds_id: int, ds_doc_id: int
    def _get_qs(self, df_docs_ids: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
        ds_doc_ids = df_docs_ids['ds_doc_id']
        ds_doc_ids = np.unique(ds_doc_ids)
        df_qrels = self.df_qrels.loc[ds_doc_ids]
        ds_qs_ids = np.unique(df_qrels['dsqid'])
        df_qs_ids = self.df_qs_ids.loc[ds_qs_ids]
        return df_qrels, df_qs_ids

    def get_embs_batch(self, doc_emb_ids: np.ndarray) -> QrelsEmbsBatch:
        df_docs_ids = self.df_docs_ids.loc[doc_emb_ids]
        offsets = doc_emb_ids * self.emb_bytes_size
        docs_embs: list[np.ndarray] = []
        for off in offsets:
            doc_emb = self.docs_embs_file.get_vec(off)
            docs_embs.append(doc_emb)
        # df_qrels, df_qs_ids = self._get_qs(df_docs_ids)

        batch = QrelsEmbsBatch(
            df_docs_ids=df_docs_ids, docs_embs=docs_embs, chunk_size=self.chunk_size, emb_size=self.emb_size,
            device=self.device,
        )
        return batch

    def get_embs_view(self, batch_size: Optional[int] = None) -> DsQrelsEmbsView:
        ids = self.df_docs_ids.index.values
        return DsQrelsEmbsView(self, ids, self.get_embs_batch, batch_size)

    def close(self):
        self.docs_embs_file.close()
        self.qs_embs_file.close()

    def __str__(self) -> str:
        return f'{self.ds_dir_path.name}. Docs embeddings: {len(self.df_docs_ids)}. Queries embeddings: {len(self.df_qs_ids)}'

    def __repr__(self) -> str:
        return str(self)


