from pathlib import Path
from typing import Optional, BinaryIO, Union, Generator

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

from mllm.data.dsqrels import DsQrels
from mllm.data.fever.dsfever import load_dsqrels_fever
from mllm.data.msmarco.dsmsmarco import load_dsqrels_msmarco
from mllm.tokenization.chunk_tokenizer import ChunkTokenizer
from mllm.train.utils import mask_random_words


def load_qrels_datasets(
        ds_dir_paths: list[Path], ch_tkz: ChunkTokenizer, emb_chunk_size: int, device: Optional[torch.device] = None,
        join: bool = True) -> Union[DsQrels, list[DsQrels]]:
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

    if not join:
        return dss

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


class HfDsIterator:
    ds: Dataset
    inds: np.ndarray
    inp_len: int
    pad_tok_ind: int
    mask_tok_repr: str
    tkz: PreTrainedTokenizer
    docs_batch_size: int
    device: torch.device
    preserve_first_token: bool

    def __init__(self, ds: Dataset, inds: np.ndarray, inp_len: int, pad_tok_ind: int, mask_tok_repr: str, tkz: PreTrainedTokenizer,
            docs_batch_size: int, device: torch.device, preserve_first_token: bool = False):
        self.ds = ds
        self.inds = inds.copy()
        self.inp_len = inp_len
        self.pad_tok_ind = pad_tok_ind
        self.mask_tok_repr = mask_tok_repr
        self.tkz = tkz
        self.docs_batch_size = docs_batch_size
        self.device = device
        self.preserve_first_token = preserve_first_token

    def get_batch_tokens(self, doc_inds: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        docs_toks = np.full((len(doc_inds), self.inp_len), self.pad_tok_ind)
        docs_toks_aug = np.full((len(doc_inds), self.inp_len), self.pad_tok_ind)
        for i, doc_ind in enumerate(doc_inds):
            doc = self.ds[int(doc_ind)]
            title, text = doc['title'], doc['text']
            if np.random.rand() < 1 / 4:
                doc_txt: str = title
            else:
                doc_txt: str = text
            # doc_txt = f'{title} {text}'
            # doc_txt = text
            doc_toks = self.tkz(doc_txt)['input_ids']
            n_toks = len(doc_toks)
            if n_toks > self.inp_len:
                if self.preserve_first_token:
                    i_off = np.random.randint(1, n_toks - self.inp_len + 1)
                    doc_toks = np.concatenate([doc_toks[:1], doc_toks[i_off:i_off + self.inp_len - 1]])
                else:
                    i_off = np.random.randint(n_toks - self.inp_len + 1)
                    doc_toks = doc_toks[i_off:i_off + self.inp_len]
            docs_toks[i, :len(doc_toks)] = doc_toks

            doc_txt_aug = mask_random_words(doc_txt, mask_tok_str=self.mask_tok_repr)
            if doc_txt_aug is None:
                doc_toks_aug = doc_toks
            else:
                doc_toks_aug = self.tkz(doc_txt_aug)['input_ids']
                n_toks_aug = len(doc_toks_aug)
                if n_toks_aug > self.inp_len:
                    if self.preserve_first_token:
                        i_off = np.random.randint(1, n_toks_aug - self.inp_len + 1)
                        doc_toks_aug = np.concatenate([doc_toks_aug[:1], doc_toks_aug[i_off:i_off + self.inp_len - 1]])
                    else:
                        i_off = np.random.randint(n_toks_aug - self.inp_len + 1)
                        doc_toks_aug = doc_toks_aug[i_off:i_off + self.inp_len]
            docs_toks_aug[i, :len(doc_toks_aug)] = doc_toks_aug

        docs_toks_t = torch.from_numpy(docs_toks).to(self.device)
        docs_toks_aug_t = torch.from_numpy(docs_toks_aug).to(self.device)
        return docs_toks_t, docs_toks_aug_t

    def get_batch(self, i_batch: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        i1 = i_batch * self.docs_batch_size
        i2 = i1 + self.docs_batch_size
        batch_inds = self.inds[i1:i2]
        rest_batch_size = self.docs_batch_size - len(batch_inds)
        if rest_batch_size > 0:
            batch_inds = batch_inds + self.inds[:rest_batch_size * self.docs_batch_size]
        if i2 >= len(batch_inds):
            i_batch = 0
            np.random.shuffle(self.inds)
        batch_toks, batch_toks_aug = self.get_batch_tokens(batch_inds)
        return batch_toks, batch_toks_aug, i_batch

    def get_batch_iterator(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        i_batch = 0
        while True:
            batch_toks, batch_toks_aug, i_batch = self.get_batch(i_batch)
            yield batch_toks, batch_toks_aug


