from pathlib import Path
from typing import Optional, BinaryIO

import numpy as np
import torch

from mllm.data.dsqrels import DsQrels
from mllm.data.fever.dsfever import load_dsqrels_fever
from mllm.data.msmarco.dsmsmarco import load_dsqrels_msmarco
from mllm.tokenization.chunk_tokenizer import ChunkTokenizer


def load_qrels_datasets(ds_dir_paths: list[Path], ch_tkz: ChunkTokenizer, emb_chunk_size: int, device: Optional[torch.device] = None) -> DsQrels:
    dss = []
    for ds_path in ds_dir_paths:
        if 'fever' in ds_path.name:
            load_fn = load_dsqrels_fever
        elif 'msmarco' in ds_path.name:
            load_fn = load_dsqrels_msmarco
        else:
            raise Exception(f'Unknown dataset: {ds_path}')
        ds = load_fn(ds_path=ds_path, ch_tkz=ch_tkz, max_chunks_per_doc=100, emb_chunk_size=emb_chunk_size, device=device)
        dss.append(ds)

    print('Join datasets:')
    for ds in dss:
        assert len(ds.ds_ids) == 1
        print(f'   {ds}')
    ds = DsQrels.join(dss)
    return ds


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
