from pathlib import Path
from typing import Optional, TextIO, Type, BinaryIO

import numpy as np
import pandas as pd
import torch

from mllm.data.common import DsView
from mllm.utils.utils import read_tsv


class QrelsDocsEmbsBatch:
    # doc_emb_id: int (index), ds_id: int, ds_doc_id: int
    df_docs_ids: pd.DataFrame
    docs_embs: list[np.ndarray]
    chunk_size: int
    emb_size: int
    device: Optional[torch.device] = None
    docs_embs_tf: Optional[torch.Tensor] = None

    def __init__(self, df_docs_ids: pd.DataFrame, docs_embs: list[np.ndarray], chunk_size: int,
                 emb_size: int, device: Optional[torch.device] = None):
        self.df_docs_ids = df_docs_ids
        self.docs_embs = docs_embs
        self.chunk_size = chunk_size
        self.emb_size = emb_size
        self.device = device

    def _to_tensor(self, arr: list[np.ndarray]) -> torch.Tensor:
        arr = np.stack(arr, axis=0)
        res = torch.from_numpy(arr)
        res = res.reshape((-1, self.chunk_size, self.emb_size))
        if self.device is not None:
            res = res.to(self.device)
        return res


    def get_tensor(self) -> torch.Tensor:
        if self.docs_embs_tf is None:
            self.docs_embs_tf = self._to_tensor(self.docs_embs)
        return self.docs_embs_tf


class DsQrelsDocsEmbsView(DsView['DsQrelsEmbs', QrelsDocsEmbsBatch]):
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
        self.fid = open(self.fpath, 'b')
        self.vec_size = vec_size
        self.dtype = dtype
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
    docs_embs_file: BinVecsFile
    qs_file: BinVecsFile
    device: Optional[torch.device] = None

    def __init__(self, ds_dir_path: Path, chunk_size: int, emb_size: int, emb_dtype: np.dtype[int], device: Optional[torch.device] = None):
        self.ds_dir_path = ds_dir_path
        self.chunk_size = chunk_size
        self.emb_size = emb_size
        self.emb_dtype = emb_dtype
        self.emb_bytes_size = self.emb_size * self.emb_dtype.itemsize
        self.device = device
        docs_ids_fpath = self.ds_dir_path / 'docs_ids.tsv'
        docs_embs_fpath = self.ds_dir_path / 'docs_embs.npy'
        qs_ids_fpath = self.ds_dir_path / 'qs_ids.tsv'
        qs_embs_fpath = self.ds_dir_path / 'qs_embs.npy'
        self.df_docs_ids = read_tsv(docs_ids_fpath)
        self.df_docs_ids.set_index('doc_emb_id', inplace=True)
        self.df_qs_ids = read_tsv(qs_ids_fpath)
        self.df_qs_ids.set_index('query_emb_id', inplace=True)
        self.docs_embs_file = BinVecsFile(fpath=docs_embs_fpath, vec_size=self.emb_size, dtype=self.emb_dtype)
        self.qs_embs_file = BinVecsFile(fpath=qs_embs_fpath, vec_size=self.emb_size, dtype=self.emb_dtype)

    def get_docs_embs_batch(self, doc_emb_ids: np.ndarray) -> QrelsDocsEmbsBatch:
        df_docs_ids = self.df_docs_ids.loc[doc_emb_ids]
        offsets = doc_emb_ids * self.emb_bytes_size
        docs_embs: list[np.ndarray] = []
        for off in offsets:
            doc_emb = self.docs_embs_file.get_vec(off)
            docs_embs.append(doc_emb)
        batch = QrelsDocsEmbsBatch(df_docs_ids=df_docs_ids, docs_embs=docs_embs, chunk_size=self.chunk_size, emb_size=self.emb_size,
                                   device=self.device)
        return batch

    def get_docs_embs_view(self, batch_size: Optional[int] = None) -> DsQrelsDocsEmbsView:
        ids = self.df_docs_ids.index.values
        return DsQrelsDocsEmbsView(self, ids, self.get_docs_embs_batch, batch_size * self.chunk_size)

    def close(self):
        self.docs_embs_file.close()
        self.qs_embs_file.close()

    def __str__(self) -> str:
        return f'{self.ds_dir_path.name}. Docs embeddings: {len(self.df_docs_ids)}. Queries embeddings: {len(self.df_qs_ids)}'

    def __repr__(self) -> str:
        return str(self)


