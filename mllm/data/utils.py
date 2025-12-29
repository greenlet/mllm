from dataclasses import dataclass
from pathlib import Path
from typing import Optional, BinaryIO, Union, Generator, Tuple, List, Any, Dict, Callable

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

from mllm.data.dsqrels import DsQrels
from mllm.data.fever.dsfever import load_dsqrels_fever
from mllm.data.msmarco.dsmsmarco import load_dsqrels_msmarco
from mllm.tokenization.chunk_tokenizer import ChunkTokenizer
from mllm.train.mask_utils import MaskCfg, mask_random_tokens, mask_random_words, mask_random_words_v2


def load_qrels_datasets(
        ds_dir_paths: list[Path], ch_tkz: Optional[ChunkTokenizer], emb_chunk_size: int, device: Optional[torch.device] = None,
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


AugTxtGen = Generator[tuple[torch.Tensor, torch.Tensor], None, None]


class HfDsIterator:
    ds: Dataset
    inds: np.ndarray
    inp_len: int
    pad_tok_ind: int
    mask_tok_repr: str
    tkz: PreTrainedTokenizer
    docs_batch_size: int
    device: torch.device
    preserve_edge_tokens: bool

    def __init__(self, ds: Dataset, inds: np.ndarray, inp_len: int, pad_tok_ind: int, mask_tok_repr: str, tkz: PreTrainedTokenizer,
                 docs_batch_size: int, device: torch.device, preserve_edge_tokens: bool = False):
        self.ds = ds
        self.inds = inds.copy()
        self.inp_len = inp_len
        self.pad_tok_ind = pad_tok_ind
        self.mask_tok_repr = mask_tok_repr
        self.tkz = tkz
        self.docs_batch_size = docs_batch_size
        self.device = device
        self.preserve_edge_tokens = preserve_edge_tokens

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
            doc_toks = np.array(doc_toks)
            n_toks = len(doc_toks)
            if n_toks > self.inp_len:
                if self.preserve_edge_tokens:
                    i_off = np.random.randint(1, n_toks - self.inp_len + 1)
                    doc_toks = np.concatenate([doc_toks[:1], doc_toks[i_off:i_off + self.inp_len - 2], doc_toks[-1:]])
                else:
                    i_off = np.random.randint(n_toks - self.inp_len + 1)
                    doc_toks = doc_toks[i_off:i_off + self.inp_len]
            docs_toks[i, :len(doc_toks)] = doc_toks

            if self.preserve_edge_tokens:
                doc_toks_aug = doc_toks.copy()
                doc_toks_aug[1:-1] = mask_random_tokens(doc_toks_aug[1:-1], self.tkz)
            else:
                doc_toks_aug = mask_random_tokens(doc_toks, self.tkz)
            docs_toks_aug[i, :len(doc_toks_aug)] = doc_toks_aug

        docs_toks_t = torch.from_numpy(docs_toks).to(self.device)
        docs_toks_aug_t = torch.from_numpy(docs_toks_aug).to(self.device)
        return docs_toks_t, docs_toks_aug_t

    def get_batch(self, i_batch: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        i1 = i_batch * self.docs_batch_size
        i2 = i1 + self.docs_batch_size
        batch_inds = self.inds[i1:i2].copy()
        rest_batch_size = self.docs_batch_size - len(batch_inds)
        if rest_batch_size > 0:
            batch_inds = np.concatenate([batch_inds, self.inds[:rest_batch_size].copy()])
        if i2 >= len(batch_inds):
            i_batch = 0
            np.random.shuffle(self.inds)
        batch_toks, batch_toks_aug = self.get_batch_tokens(batch_inds)
        return batch_toks, batch_toks_aug, i_batch

    def get_batch_iterator(self) -> AugTxtGen:
        i_batch = 0
        while True:
            batch_toks, batch_toks_aug, i_batch = self.get_batch(i_batch)
            yield batch_toks, batch_toks_aug


def split_df(df: pd.DataFrame, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_total = len(df)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    return df.iloc[:n_train], df.iloc[n_train:]


@dataclass(kw_only=True)
class TokensSubset:
    toks_src: list[int]
    inp_beg_ind: int
    inp_end_ind: int
    toks_inp: list[int]
    cite_beg_ind: int = -1
    cite_end_ind: int = -1
    toks_cite: Optional[list[int]] = None


class RandomInputTokenizer:
    def __init__(self, tkz: PreTrainedTokenizer, max_len: int):
        self.tkz = tkz
        self.cite_beg_word, self.cite_end_word = '<cite>', '</cite>'
        self.cite_beg_toks, self.cite_end_toks = self.tkz([self.cite_beg_word, self.cite_end_word], add_special_tokens=False).input_ids
        self.max_len = max_len
        self.n_inp_toks = self.max_len - 2  # -2 for CLS and SEP
        self.n_cite_toks = self.n_inp_toks - len(self.cite_beg_toks) - len(self.cite_end_toks)
        assert self.n_cite_toks > 0, f'max_len={self.max_len} must be greater then total size: {self.cite_beg_word}={len(self.cite_beg_toks)} + ' \
            f'{self.cite_end_word}={len(self.cite_end_toks)} + {self.tkz.cls_token}=1 + {self.tkz.sep_token}=1'
    
    def __call__(self, texts: list[str], n_items_to_cite: int = 1) -> List[TokensSubset]:
        batch_size = len(texts)
        input_ids = self.tkz(texts, add_special_tokens=False).input_ids
        batch: list[TokensSubset] = []
        cite_inds = None
        if n_items_to_cite > 0:
            cite_inds = np.random.choice(batch_size, size=n_items_to_cite, replace=False)

        for i in range(batch_size):
            to_cite = (cite_inds is not None) and (i in cite_inds) or (n_items_to_cite < 0)
            cur_len = len(input_ids[i])
            n_inp_toks = self.n_inp_toks if not to_cite else self.n_cite_toks
            if cur_len <= n_inp_toks:
                inp_beg_ind = 0
                inp_end_ind = cur_len
            else:
                max_beg_ind = cur_len - n_inp_toks
                inp_beg_ind = np.random.randint(0, max_beg_ind + 1)
                inp_end_ind = inp_beg_ind + n_inp_toks
            toks_inp = input_ids[i][inp_beg_ind:inp_end_ind]
            cite_beg_ind, cite_end_ind, toks_cite = -1, -1, None
            if to_cite:
                n_cite_toks = np.random.randint(1, len(toks_inp) + 1)
                sub_off = np.random.randint(0, len(toks_inp) - n_cite_toks + 1)
                toks_inp = [self.tkz.cls_token_id] + toks_inp[:sub_off] + self.cite_beg_toks + \
                    toks_inp[sub_off:sub_off + n_cite_toks] + self.cite_end_toks + \
                    toks_inp[sub_off + n_cite_toks:] + [self.tkz.sep_token_id]
                cite_beg_ind = inp_beg_ind + sub_off
                cite_end_ind = cite_beg_ind + n_cite_toks
                toks_cite = input_ids[i][cite_beg_ind:cite_end_ind]
            else:
                toks_inp = [self.tkz.cls_token_id, *toks_inp, self.tkz.sep_token_id]
            toks_sub = TokensSubset(
                toks_src=input_ids[i],
                inp_beg_ind=inp_beg_ind,
                inp_end_ind=inp_end_ind,
                toks_inp=toks_inp,
                cite_beg_ind=cite_beg_ind,
                cite_end_ind=cite_end_ind,
                toks_cite=toks_cite,
            )
            batch.append(toks_sub)
        return batch


@dataclass(kw_only=True)
class TokensSubsetV2:
    toks_src: list[int]
    inp_beg_ind: int
    inp_end_ind: int
    toks_inp: list[int]
    cite_beg_ind: int
    cite_end_ind: int
    toks_cite_masked: list[int]
    toks_cite: list[int]
    toks_cite_beg: list[int]
    toks_cite_end: list[int]
    prompt: str
    toks_prompt: list[int]


class RandomInputTokenizerV2:
    def __init__(self, tkz: PreTrainedTokenizer, max_len: int, n_random_toks: int, n_special_toks: int = 1000, mask_cfg: Optional[MaskCfg] = None):
        self.tkz = tkz
        self.max_len = max_len
        self.n_random_toks = n_random_toks
        self.n_special_toks = n_special_toks
        self.shuffled_tok_ids = np.random.permutation(np.arange(self.n_special_toks, len(tkz)))  # Exclude special tokens
        self.shuffled_tok_cur_ind = 0
        self.n_inp_toks = self.max_len - 2  # -2 for CLS and SEP
        self.n_cite_toks = self.n_inp_toks - 2 * self.n_random_toks
        assert self.n_cite_toks > 0, f'max_len={self.max_len} must be greater then total size: n_random_toks*2={self.n_random_toks * 2} + ' \
            f'{self.tkz.cls_token}=1 + {self.tkz.sep_token}=1'
        self.prompt_template = 'Cite tag begin: "{}". Cite tag end: "{}". Produce output text between these tags.'
        prompt = self.prompt_template.format('a b c', 'd e f')
        toks_prompt = self.tkz(prompt, add_special_tokens=True).input_ids
        assert len(toks_prompt) < self.max_len, f'Prompt length {len(toks_prompt)} must be less than max_len={self.max_len}. Prompt template: {self.prompt_template} ' \
            f'(example: {prompt}) = {len(toks_prompt)} tokens >= max tokens = {self.max_len})'
        self.mask_cfg = mask_cfg

    
    def _next_random_tokens(self) -> list[int]:
        toks = self.shuffled_tok_ids[self.shuffled_tok_cur_ind:self.shuffled_tok_cur_ind + self.n_random_toks].tolist()
        self.shuffled_tok_cur_ind += len(toks)
        if self.shuffled_tok_cur_ind == len(self.shuffled_tok_ids):
            if len(toks) < self.n_random_toks:
                n = self.n_random_toks - len(toks)
                toks += self.shuffled_tok_ids[:n].tolist()
            np.random.shuffle(self.shuffled_tok_ids)
            self.shuffled_tok_cur_ind = 0
        return toks

    
    def __call__(self, texts: list[str]) -> List[TokensSubsetV2]:
        batch_size = len(texts)
        input_ids = self.tkz(texts, add_special_tokens=False).input_ids
        batch: list[TokensSubsetV2] = []

        for i in range(batch_size):
            cur_len = len(input_ids[i])
            n_inp_toks = self.n_cite_toks
            if cur_len <= n_inp_toks:
                inp_beg_ind = 0
                inp_end_ind = cur_len
            else:
                max_beg_ind = cur_len - n_inp_toks
                inp_beg_ind = np.random.randint(0, max_beg_ind + 1)
                inp_end_ind = inp_beg_ind + n_inp_toks
            toks_inp = input_ids[i][inp_beg_ind:inp_end_ind]

            n_cite_toks = np.random.randint(1, len(toks_inp) + 1)
            sub_off = np.random.randint(0, len(toks_inp) - n_cite_toks + 1)
            toks_cite_beg = self._next_random_tokens()
            toks_cite_end = self._next_random_tokens()
            toks_cite = input_ids[i][cite_beg_ind:cite_end_ind]

            toks_cite_masked, _ = mask_random_words_v2(np.array(toks_cite), self.tkz, self.mask_cfg)
            toks_cite_masked = toks_cite_masked.tolist()
            
            toks_inp = [self.tkz.cls_token_id] + toks_inp[:sub_off] + toks_cite_beg + \
                toks_cite_masked + toks_cite_end + \
                toks_inp[sub_off + n_cite_toks:] + [self.tkz.sep_token_id]
            cite_beg_ind = inp_beg_ind + sub_off
            cite_end_ind = cite_beg_ind + n_cite_toks

            prompt = self.prompt_template.format(
                ' '.join(self.tkz.convert_ids_to_tokens(toks_cite_beg)),
                ' '.join(self.tkz.convert_ids_to_tokens(toks_cite_end))
            )
            toks_prompt = self.tkz(prompt, add_special_tokens=True).input_ids

            toks_sub = TokensSubsetV2(
                toks_src=input_ids[i],
                inp_beg_ind=inp_beg_ind,
                inp_end_ind=inp_end_ind,
                toks_inp=toks_inp,
                cite_beg_ind=cite_beg_ind,
                cite_end_ind=cite_end_ind,
                toks_cite_masked=toks_cite_masked,
                toks_cite=toks_cite,
                toks_cite_beg=toks_cite_beg,
                toks_cite_end=toks_cite_end,
                prompt=prompt,
                toks_prompt=toks_prompt,
            )
            batch.append(toks_sub)
        return batch


def tokens_subsets_to_tensors(batch: List[TokensSubset], pad_token_id: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = torch.device('cpu')
    batch_size = len(batch)
    max_len = max(len(item.toks_inp) for item in batch)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for i in range(batch_size):
        cur_len = len(batch[i].toks_inp)
        input_ids[i, :cur_len] = torch.tensor(batch[i].toks_inp, dtype=torch.long, device=device)
        attention_mask[i, :cur_len] = 1
    return input_ids, attention_mask


def tokens_subsets_v2_to_tensors(batch: List[TokensSubsetV2], tkz: PreTrainedTokenizer, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if device is None:
        device = torch.device('cpu')
    batch_size = len(batch)
    max_len = max(max(len(item.toks_inp), len(item.toks_prompt)) for item in batch)
    input_ids = torch.full((batch_size * 2, max_len), tkz.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size * 2, max_len), dtype=torch.long, device=device)
    for i in range(batch_size):
        toks_inp_len, toks_prompt_len = len(batch[i].toks_inp), len(batch[i].toks_prompt)
        input_ids[i, :toks_inp_len] = torch.tensor(batch[i].toks_inp, dtype=torch.long, device=device)
        attention_mask[i, :toks_inp_len] = 1
        input_ids[batch_size + i, :toks_prompt_len] = torch.tensor(batch[i].toks_prompt, dtype=torch.long, device=device)
        attention_mask[batch_size + i, :toks_prompt_len] = 1

    edge_inds = torch.stack([
        torch.arange(batch_size, device=device),
        torch.full((batch_size,), batch_size, device=device),
    ])

    return input_ids, attention_mask, edge_inds


