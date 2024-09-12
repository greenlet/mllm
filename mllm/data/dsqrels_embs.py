from pathlib import Path
from typing import Optional, TextIO, Type, BinaryIO

import numpy as np
import pandas as pd
import torch

from mllm.utils.utils import read_tsv


class DsQrelsEmbsView:
    ds: 'DsQrelsEmbs'
    ids: np.ndarray
    batch_size: Optional[int] = None


class BinVecsFile:
    fpath: Path
    fid: BinaryIO
    vec_size: int
    dtype: np.dtype
    bytes_size: int
    opened: bool

    def __init__(self, fpath: Path, vec_size: int, dtype: Type):
        self.fpath = fpath
        self.fid = open(self.fpath, 'b')
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
    embs_size: int
    embs_dtype: Type
    df_docs_ids: pd.DataFrame
    df_qs_ids: pd.DataFrame
    docs_embs_file: BinVecsFile
    qs_file: BinVecsFile
    device: Optional[torch.device] = None

    def __init__(self, ds_dir_path: Path, chunk_size: int, embs_size: int, embs_dtype: Type, device: Optional[torch.device] = None):
        self.ds_dir_path = ds_dir_path
        self.chunk_size = chunk_size
        self.embs_size = embs_size
        self.embs_dtype = embs_dtype
        self.device = device
        docs_ids_fpath = self.ds_dir_path / 'docs_ids.tsv'
        docs_embs_fpath = self.ds_dir_path / 'docs_embs.npy'
        qs_ids_fpath = self.ds_dir_path / 'qs_ids.tsv'
        qs_embs_fpath = self.ds_dir_path / 'qs_embs.npy'
        self.df_docs_ids = read_tsv(docs_ids_fpath)
        self.df_qs_ids = read_tsv(qs_ids_fpath)
        self.docs_embs_file = BinVecsFile(fpath=docs_embs_fpath, vec_size=self.embs_size, dtype=self.embs_dtype)
        self.qs_embs_file = BinVecsFile(fpath=qs_embs_fpath, vec_size=self.embs_size, dtype=self.embs_dtype)

    def close(self):
        self.docs_embs_file.close()
        self.qs_embs_file.close()

