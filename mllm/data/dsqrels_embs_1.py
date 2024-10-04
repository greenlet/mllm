from pathlib import Path
from typing import Optional, TextIO, Type, BinaryIO, Union

import numpy as np
import pandas as pd
import torch

from mllm.data.common import DsView
from mllm.data.utils import BinVecsFile
from mllm.utils.utils import read_tsv


class QrelsEmbs1Batch:

    def __init__(self):
        pass

class DsQrelsEmbs1View(DsView['DsQrelsEmbs1', QrelsEmbs1Batch]):
    pass


class DsQrelsEmbs1:
    embs_ds_dir_path: Path
    embs_1_ds_dir_path: Path
    chunk_size: int
    emb_size: int
    emb_dtype: np.dtype
    emb_bytes_size: int
    # doc_emb_id: int (index), ds_id: int, ds_doc_id: int
    df_docs_ids: pd.DataFrame
    # query_emb_id: int (index), ds_id: int, ds_query_id: int
    df_qs_ids: pd.DataFrame
    # qid: int, did: int, dsqid: int (generated), dsdid: int (generated)
    df_qrels: pd.DataFrame
    df_docs_embs_1_ids: pd.DataFrame
    qs_embs_file: BinVecsFile
    docs_embs_1_file: BinVecsFile
    device: Optional[torch.device] = None

    def __init__(self, embs_ds_dir_path: Path, embs_1_ds_dir_path: Path, emb_size: int, emb_dtype: np.dtype, device: Optional[torch.device] = None):
        self.embs_ds_dir_path = embs_ds_dir_path
        self.embs_1_ds_dir_path = embs_1_ds_dir_path
        self.emb_size = emb_size
        self.emb_dtype = emb_dtype
        self.emb_bytes_size = self.emb_size * np.dtype(self.emb_dtype).itemsize
        self.device = device

        docs_ids_fpath = self.embs_ds_dir_path / 'docs_ids.tsv'
        qs_ids_fpath = self.embs_ds_dir_path / 'qs_ids.tsv'
        qs_embs_fpath = self.embs_ds_dir_path / 'qs_embs.npy'
        qrels_fpath = self.embs_ds_dir_path / 'qrels.tsv'
        docs_embs_1_ids_fpath = self.embs_1_ds_dir_path / 'docs_embs_ids.tsv'
        docs_embs_1_fpath = self.embs_1_ds_dir_path / 'docs_embs.npy'

        self.df_docs_ids = read_tsv(docs_ids_fpath)
        self.df_qs_ids = read_tsv(qs_ids_fpath)
        self.df_qs_ids.set_index('ds_query_id', inplace=True)
        self.df_qrels = read_tsv(qrels_fpath)
        self.df_qrels.set_index('dsdid', inplace=True)
        self.df_docs_embs_1_ids = read_tsv(docs_embs_1_ids_fpath)
        self.qs_embs_file = BinVecsFile(fpath=qs_embs_fpath, vec_size=self.emb_size, dtype=self.emb_dtype)
        self.docs_embs_1_file = BinVecsFile(fpath=docs_embs_1_fpath, vec_size=self.emb_size, dtype=self.emb_dtype)

    # doc_emb_id: int (index), ds_id: int, ds_doc_id: int
    # def _get_qs(self, df_docs_ids: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
    #     ds_doc_ids = df_docs_ids['ds_doc_id']
    #     ds_doc_ids = np.unique(ds_doc_ids)
    #     df_qrels = self.df_qrels.loc[ds_doc_ids]
    #     ds_qs_ids = np.unique(df_qrels['dsqid'])
    #     df_qs_ids = self.df_qs_ids.loc[ds_qs_ids]
    #     return df_qrels, df_qs_ids
    #
    # def get_embs_batch(self, doc_emb_ids: np.ndarray) -> QrelsEmbs1Batch:
    #     df_docs_ids = self.df_docs_ids.loc[doc_emb_ids]
    #     offsets = doc_emb_ids * self.emb_bytes_size
    #     docs_embs: list[np.ndarray] = []
    #     for off in offsets:
    #         doc_emb = self.docs_embs_file.get_vec(off)
    #         docs_embs.append(doc_emb)
    #     # df_qrels, df_qs_ids = self._get_qs(df_docs_ids)
    #
    #     batch = QrelsEmbs1Batch(
    #         df_docs_ids=df_docs_ids, docs_embs=docs_embs, chunk_size=self.chunk_size, emb_size=self.emb_size,
    #         device=self.device,
    #     )
    #     return batch

