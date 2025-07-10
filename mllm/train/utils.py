import os.path
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Generator, Any

import numpy as np
import pandas as pd
import torch
import torch.utils.tensorboard as tb
from datasets import Dataset, load_dataset
from torch import nn
from torch.nn.modules import activation
from transformers import PreTrainedTokenizer, AutoTokenizer

from mllm.data.common import DsView, TDs, TBatch
from mllm.data.wiki.itwiki import get_wiki_iterators
from mllm.train.mask_utils import mask_random_tokens, mask_random_words
from mllm.utils.utils import gen_dt_str, DT_PAT_RE, parse_dt_str

SUBDIR_PAT_STR = re.compile(r'^[\w-]*?-(%s)-.*$' % DT_PAT_RE)
SUBDIR_PAT = re.compile(SUBDIR_PAT_STR)
DT_PAT = re.compile(DT_PAT_RE)


def get_dt_from_subdir(subdir: str) -> Optional[str]:
    parts = subdir.split('-')
    for part in parts:
        if DT_PAT.match(part):
            return part


def gen_train_subdir(prefix: Optional[str], postfix: Optional[str]) -> str:
    subdir = gen_dt_str()
    if prefix:
        subdir = f'{prefix}-{subdir}'
    if postfix:
        subdir = f'{subdir}-{postfix}'
    return subdir


def find_last_train_subdir(train_root_path: Path, prefix: Optional[str] = None, postfix: Optional[str] = None) -> Optional[Path]:
    dt_last: Optional[datetime] = None
    subdir_last: Optional[str] = None
    for subpath in train_root_path.iterdir():
        if not subpath.is_dir():
            continue
        subdir = subpath.name
        if prefix:
            # print(subdir, subdir.startswith(prefix))
            if not subdir.startswith(prefix):
                continue
            subdir = subdir[len(prefix):]
        if postfix:
            # print(subdir, subdir.endswith(postfix), postfix)
            if not subdir.endswith(postfix):
                continue
            subdir = subdir[:-len(postfix)]
        assert subdir, f'prefix: {prefix}. postfix: {postfix}. subdir: {subpath.name}'
        print(subdir)
        m = SUBDIR_PAT.match(subdir)
        if not m:
            continue
        dt_cur = parse_dt_str(m.group(1))
        if dt_cur is None:
            continue
        if dt_last is None or dt_cur > dt_last:
            dt_last = dt_cur
            subdir_last = subpath.name
    if subdir_last is not None:
        return train_root_path / subdir_last


def find_create_train_path(train_root_path: Path, prefix: Optional[str] = None, postfix: Optional[str] = None, subdir: Optional[str] = None) -> Path:
    if subdir == 'last':
        train_path = find_last_train_subdir(train_root_path, prefix, postfix)
        if train_path is None:
            raise Exception(f'Cannot find last subdirectory of the format `{SUBDIR_PAT_STR}` (prefix = {prefix}, postfix = {postfix}) in {train_root_path}')
    elif subdir:
        train_path = train_root_path / subdir
        assert train_path.exists(), f'Directory {train_path} does not exist'
    else:
        train_subdir = gen_train_subdir(prefix, postfix)
        train_path = train_root_path / train_subdir
        train_path.mkdir(parents=True, exist_ok=True)
    return train_path


def print_grad(model: torch.nn.Module):
    for name, p in model.named_parameters():
        grad = p.grad.cpu().detach().numpy()
        p = p.cpu().detach().numpy()
        eps = 1e-8
        print(name, p.dtype, grad.shape, np.prod(list(grad.shape)), (grad < eps).sum())
        print(' ' * 4, p.min(), p.mean(), p.max())


def calc_print_batches(view_train: DsView[TDs, TBatch], view_val: DsView[TDs, TBatch], batch_size: int, items_name: str) -> tuple[int, int]:
    calc_batches = lambda n_items: n_items // batch_size + (n_items % batch_size > 1)
    n_qs_train, n_qs_val = len(view_train), len(view_val)
    n_batches_train = calc_batches(n_qs_train)
    n_batches_val = calc_batches(n_qs_val)
    print(f'{items_name} train: {n_qs_train}')
    print(f'{items_name} val: {n_qs_val}')
    print(f'Batches train: {n_batches_train}')
    print(f'Batches val: {n_batches_val}')
    return n_batches_train, n_batches_val


def calc_print_batches_multi(views_train: list[DsView[TDs, TBatch]], views_val: list[DsView[TDs, TBatch]], batch_size: int, items_name: str) -> tuple[int, int]:
    calc_batches = lambda n_items: n_items // batch_size + (n_items % batch_size > 1)
    n_qs_train, n_qs_val = sum(len(v) for v in views_train), sum(len(v) for v in views_val)
    n_batches_train = calc_batches(n_qs_train)
    n_batches_val = calc_batches(n_qs_val)
    print(f'{items_name} train: {n_qs_train}')
    print(f'{items_name} val: {n_qs_val}')
    print(f'Batches train: {n_batches_train}')
    print(f'Batches val: {n_batches_val}')
    return n_batches_train, n_batches_val


def concat_tokens(*chunks: torch.Tensor, shuffle: bool = True) ->torch.Tensor:
    if shuffle:
        chunks = list(chunks)
        np.random.shuffle(chunks)
    return torch.concat(chunks, dim=0)


# chunks: input token chunks of the shape [n_docs, n_tokens_per_doc]
def remove_tokens(chunks: torch.Tensor, mask_tok: int, rem_ratio: float = 0.15, rem_conseq_ratio: float = 0.3) -> torch.Tensor:
    res = chunks.clone()
    rv = np.random.rand()
    if rv < 1 / 3:
        p = rem_ratio
        mask = torch.distributions.Bernoulli(probs=p).sample(chunks.size()).to(chunks.device)
        res[mask.bool()] = mask_tok
    elif rv < 2 / 3:
        n = chunks.shape[-1]
        n_rem = int(n * rem_conseq_ratio)
        n_rem = np.random.randint(1, n_rem)
        i = np.random.randint(n - n_rem + 1)
        res[:, i:i + n_rem] = mask_tok
    return res


def calc_params_grads_stats(params: torch.nn.Parameter) -> tuple[float, Optional[float], Optional[float], Optional[float]]:
    param_mean = params.mean().detach().cpu().item()
    param_std = None
    if params.numel() > 1:
        param_std = params.std().detach().cpu().item()
    grad_mean, grad_std = None, None
    if params.grad is not None:
        grad_mean = params.grad.mean().detach().cpu().item()
        if params.grad.numel() > 1:
            grad_std = params.grad.std().detach().cpu().item()
    return param_mean, param_std, grad_mean, grad_std


def log_weights_grads_stats(step: int, model: torch.nn.Module, tbsw: tb.SummaryWriter):
    for i, (pname, params) in enumerate(model.named_parameters()):
        pname = f'{i:02d}-{pname}'
        weight_mean, weight_std, grad_mean, grad_std = calc_params_grads_stats(params)
        tbsw.add_scalar(f'{pname}/WeightMean', weight_mean, step)
        if weight_std is not None:
            tbsw.add_scalar(f'{pname}/WeightStd', weight_std, step)
        if grad_mean is not None:
            tbsw.add_scalar(f'{pname}/GradMean', grad_mean, step)
        if grad_std is not None:
            tbsw.add_scalar(f'{pname}/GradStd', grad_std, step)


Activation = Callable[..., nn.Module]


def get_activation_module(act: str) -> Activation:
    # get list from activation submodule as lower-case
    activations_lc = [str(a).lower() for a in activation.__all__]
    if (act := str(act).lower()) in activations_lc:
        # match actual name from lower-case list, return function/factory
        idx = activations_lc.index(act)
        act_name = activation.__all__[idx]
        act_func = getattr(activation, act_name)
        return act_func
    else:
        raise ValueError(f'Cannot find activation function for string <{act}>')


ChunkTargetToksGen = Generator[tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Any], None, None]


class EedWikiIterator:
    ds: Dataset
    inds: np.ndarray
    inp_len: int
    pad_tok_ind: int
    mask_tok_repr: str
    tkz: PreTrainedTokenizer
    docs_batch_size: int
    device: torch.device
    preserve_edge_tokens: bool
    conseq: bool = False
    rem_freq: float = 0
    rem_conseq_freq: float = 1
    rem_conseq_max_len: int = 30
    rem_conseq_max_times: int = 1

    def __init__(self, ds: Dataset, inds: np.ndarray, inp_len: int, tkz: PreTrainedTokenizer,
                 docs_batch_size: int, device: torch.device, preserve_edge_tokens: bool = False, conseq: bool = False):
        assert tkz.pad_token_id is not None
        self.ds = ds
        self.inds = inds.copy()
        self.inp_len = inp_len
        self.pad_tok_ind = tkz.pad_token_id
        self.mask_tok_repr = tkz.mask_token
        self.tkz = tkz
        self.docs_batch_size = docs_batch_size
        self.device = device
        self.preserve_edge_tokens = preserve_edge_tokens
        self.conseq = conseq

    def mask_tokens(self, toks: np.ndarray) -> np.ndarray:
        if self.conseq:
            n_toks = len(toks)
            assert n_toks > 1, f'n_toks (={n_toks}) must be > 1'
            max_prob, max_len = 0.2, 15
            max_len = min(max_len, int(max_prob * len(toks)), n_toks // 2)
            max_len = max(max_len, 1)
            mask_len = np.random.randint(1, max_len + 1)
            mask_off = np.random.randint(n_toks - mask_len + 1)
            res = toks.copy()
            res[mask_off:mask_off + mask_len] = self.tkz.mask_token_id
        else:
            res = mask_random_tokens(
                toks, self.tkz, rem_freq=self.rem_freq, rem_conseq_freq=self.rem_conseq_freq,
                rem_conseq_max_len=self.rem_conseq_max_len, rem_conseq_max_times=self.rem_conseq_max_times,
            )
        return res

    def get_batch_tokens(self, doc_inds: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        docs_toks_src = np.full((len(doc_inds), self.inp_len), self.pad_tok_ind)
        docs_toks_aug = np.full((len(doc_inds), self.inp_len), self.pad_tok_ind)
        docs_toks_tgt = None
        i_rnd = np.random.randint(len(doc_inds))
        for i, doc_ind in enumerate(doc_inds):
            doc = self.ds[int(doc_ind)]
            title, text = doc['title'], doc['text']
            # if np.random.rand() < 1 / 4:
            #     doc_txt: str = title
            # else:
            #     doc_txt: str = text
            # doc_txt = f'{title} {text}'
            doc_txt = text
            doc_toks = self.tkz(doc_txt)['input_ids']
            doc_toks = np.array(doc_toks)
            n_toks = len(doc_toks)
            if n_toks > self.inp_len:
                if self.preserve_edge_tokens:
                    i_off = np.random.randint(1, n_toks - self.inp_len + 1)
                    doc_toks = np.concatenate([doc_toks[:1], doc_toks[i_off:i_off + self.inp_len - 2], doc_toks[-1:]])
                else:
                    i_off = np.random.randint(n_toks - self.inp_len + 1)
                    doc_toks = doc_toks[i_off:i_off + self.inp_len].copy()
            docs_toks_src[i, :len(doc_toks)] = doc_toks

            if i == i_rnd:
                toks_tgt = doc_toks.copy()
                if self.preserve_edge_tokens:
                    doc_toks[1:-1] = self.mask_tokens(doc_toks[1:-1])
                else:
                    doc_toks = self.mask_tokens(doc_toks)
                docs_toks_tgt = np.concatenate([toks_tgt[doc_toks == self.tkz.mask_token_id], [self.tkz.sep_token_id]])
                # print(i, self.tkz.decode(docs_toks_tgt))

            docs_toks_aug[i, :len(doc_toks)] = doc_toks

        docs_toks_src_t = torch.from_numpy(docs_toks_src).to(self.device)
        docs_toks_aug_t = torch.from_numpy(docs_toks_aug).to(self.device)
        docs_toks_tgt_t = torch.from_numpy(docs_toks_tgt).to(self.device)
        return docs_toks_aug_t, docs_toks_src_t, docs_toks_tgt_t

    def get_batch(self, i_batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        i1 = i_batch * self.docs_batch_size
        i2 = i1 + self.docs_batch_size
        batch_inds = self.inds[i1:i2].copy()
        rest_batch_size = self.docs_batch_size - len(batch_inds)
        if rest_batch_size > 0:
            batch_inds = np.concatenate([batch_inds, self.inds[:rest_batch_size].copy()])
        if i2 >= len(batch_inds):
            i_batch = 0
            np.random.shuffle(self.inds)
        batch_toks_aug, batch_toks, batch_toks_tgt = self.get_batch_tokens(batch_inds)
        return batch_toks_aug, batch_toks, batch_toks_tgt, i_batch

    def get_batch_iterator(self) -> ChunkTargetToksGen:
        i_batch = 0
        while True:
            batch_toks_aug, batch_toks, batch_toks_tgt, i_batch = self.get_batch(i_batch)
            yield batch_toks_aug, batch_toks, batch_toks_tgt, None


def get_wiki_ds_batch_iterators(
        wiki_ds_name: str, data_path: Path, inp_len: int, docs_batch_size: int, tkz: PreTrainedTokenizer, mask_conseq: bool,
        device: torch.device, shuffle: bool = False, val_ratio: float = 0.05) -> tuple[ChunkTargetToksGen, ChunkTargetToksGen]:
    print(f'Loading Wikipedia dataset: {wiki_ds_name}')
    wiki_ds_subdir = 'wikipedia'
    # dss = load_dataset(wiki_ds_subdir, wiki_ds_name, beam_runner='DirectRunner', cache_dir=str(data_path))
    dss = load_dataset(wiki_ds_subdir, wiki_ds_name, cache_dir=str(data_path))
    ds = dss['train']
    n_docs = len(ds)
    print(f'Wikipedia {wiki_ds_name} docs: {n_docs}')

    doc_inds = np.arange(n_docs)
    # np.random.seed(777)
    np.random.shuffle(doc_inds)
    n_docs_val = int(n_docs * val_ratio)
    n_docs_train = n_docs - n_docs_val
    doc_inds_train, doc_inds_val = doc_inds[:n_docs_train].copy(), doc_inds[n_docs_train:].copy()

    if shuffle:
        np.random.shuffle(doc_inds_train)
        np.random.shuffle(doc_inds_val)

    train_batch_it = EedWikiIterator(
        ds=ds, inds=doc_inds_train, inp_len=inp_len, tkz=tkz, docs_batch_size=docs_batch_size, device=device,
        preserve_edge_tokens=True, conseq=mask_conseq,
    ).get_batch_iterator()
    val_batch_it = EedWikiIterator(
        ds=ds, inds=doc_inds_val, inp_len=inp_len, tkz=tkz, docs_batch_size=docs_batch_size, device=device,
        preserve_edge_tokens=True, conseq=mask_conseq,
    ).get_batch_iterator()
    return train_batch_it, val_batch_it



class EedWikiIterator2:
    ds: Dataset
    inds: np.ndarray
    inp_len: int
    pad_tok_ind: int
    mask_tok_repr: str
    tkz: PreTrainedTokenizer
    docs_batch_size: int
    device: torch.device
    preserve_edge_tokens: bool
    conseq: bool = False
    rem_freq: float = 0
    rem_conseq_freq: float = 1
    rem_conseq_max_len: int = 30
    rem_conseq_max_times: int = 1

    def __init__(
            self, ds: Dataset, inds: np.ndarray, inp_len: int, tkz: PreTrainedTokenizer,
            docs_batch_size: int, device: torch.device, preserve_edge_tokens: bool = False, conseq: bool = False,
        ):
        assert tkz.pad_token_id is not None
        self.ds = ds
        self.inds = inds.copy()
        self.inp_len = inp_len
        self.pad_tok_ind = tkz.pad_token_id
        self.mask_tok_repr = tkz.mask_token
        self.tkz = tkz
        self.docs_batch_size = docs_batch_size
        self.device = device
        self.preserve_edge_tokens = preserve_edge_tokens
        self.conseq = conseq

    def mask_tokens(self, toks: np.ndarray) -> np.ndarray:
        if self.conseq:
            n_toks = len(toks)
            assert n_toks > 1, f'n_toks (={n_toks}) must be > 1'
            max_prob, max_len = 0.2, 15
            max_len = min(max_len, int(max_prob * len(toks)), n_toks // 2)
            max_len = max(max_len, 1)
            mask_len = np.random.randint(1, max_len + 1)
            mask_off = np.random.randint(n_toks - mask_len + 1)
            res = toks.copy()
            res[mask_off:mask_off + mask_len] = self.tkz.mask_token_id
        else:
            res = mask_random_tokens(
                toks, self.tkz, rem_freq=self.rem_freq, rem_conseq_freq=self.rem_conseq_freq,
                rem_conseq_max_len=self.rem_conseq_max_len, rem_conseq_max_times=self.rem_conseq_max_times,
            )
        return res

    def get_batch_tokens(self, doc_inds: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_docs = len(doc_inds)
        res_shape = (n_docs, self.inp_len)
        docs_toks_src = np.full(res_shape, self.pad_tok_ind)
        docs_toks_aug = np.full(res_shape, self.pad_tok_ind)
        docs_toks_tgt = np.full(res_shape, self.pad_tok_ind)
        for i, doc_ind in enumerate(doc_inds):
            doc = self.ds[int(doc_ind)]
            title, text = doc['title'], doc['text']
            # if np.random.rand() < 1 / 4:
            #     doc_txt: str = title
            # else:
            #     doc_txt: str = text
            # doc_txt = f'{title} {text}'
            doc_txt = text
            doc_toks = self.tkz(doc_txt)['input_ids']
            if len(doc_toks) <= 3:
                doc_txt = f'{title} {text}'
                doc_toks = self.tkz(doc_txt)['input_ids']
            doc_toks = np.array(doc_toks)
            n_toks = len(doc_toks)
            if n_toks > self.inp_len:
                if self.preserve_edge_tokens:
                    i_off = np.random.randint(1, n_toks - self.inp_len + 1)
                    doc_toks = np.concatenate([doc_toks[:1], doc_toks[i_off:i_off + self.inp_len - 2], doc_toks[-1:]])
                else:
                    i_off = np.random.randint(n_toks - self.inp_len + 1)
                    doc_toks = doc_toks[i_off:i_off + self.inp_len].copy()
            docs_toks_src[i, :len(doc_toks)] = doc_toks

            toks_tgt = doc_toks.copy()
            if self.preserve_edge_tokens:
                doc_toks[1:-1] = self.mask_tokens(doc_toks[1:-1])
            else:
                doc_toks = self.mask_tokens(doc_toks)
            tgt_mask = doc_toks == self.tkz.mask_token_id
            doc_toks_tgt = toks_tgt[tgt_mask]

            docs_toks_aug[i, :len(doc_toks)] = doc_toks
            docs_toks_tgt[i, :len(doc_toks_tgt)] = doc_toks_tgt

        docs_toks_src_t = torch.from_numpy(docs_toks_src).to(self.device)
        docs_toks_aug_t = torch.from_numpy(docs_toks_aug).to(self.device)
        docs_toks_tgt_t = torch.from_numpy(docs_toks_tgt).to(self.device)
        return docs_toks_aug_t, docs_toks_src_t, docs_toks_tgt_t

    def get_batch(self, i_batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        i1 = i_batch * self.docs_batch_size
        i2 = i1 + self.docs_batch_size
        batch_inds = self.inds[i1:i2].copy()
        rest_batch_size = self.docs_batch_size - len(batch_inds)
        if rest_batch_size > 0:
            batch_inds = np.concatenate([batch_inds, self.inds[:rest_batch_size].copy()])
        if i2 >= len(batch_inds):
            i_batch = 0
            np.random.shuffle(self.inds)
        batch_toks_aug, batch_toks, batch_toks_tgt = self.get_batch_tokens(batch_inds)
        return batch_toks_aug, batch_toks, batch_toks_tgt, i_batch

    def get_batch_iterator(self) -> ChunkTargetToksGen:
        i_batch = 0
        while True:
            batch_toks_aug, batch_toks, batch_toks_tgt, i_batch = self.get_batch(i_batch)
            yield batch_toks_aug, batch_toks, batch_toks_tgt, None


def get_wiki_ds_batch_iterators2(
        wiki_ds_name: str, data_path: Path, inp_len: int, docs_batch_size: int, tkz: PreTrainedTokenizer, mask_conseq: bool,
        device: torch.device, shuffle: bool = False, val_ratio: float = 0.05) -> tuple[ChunkTargetToksGen, ChunkTargetToksGen]:
    print(f'Loading Wikipedia dataset: {wiki_ds_name}')
    wiki_ds_subdir = 'wikipedia'
    # dss = load_dataset(wiki_ds_subdir, wiki_ds_name, beam_runner='DirectRunner', cache_dir=str(data_path))
    dss = load_dataset(wiki_ds_subdir, wiki_ds_name, cache_dir=str(data_path))
    ds = dss['train']
    n_docs = len(ds)
    print(f'Wikipedia {wiki_ds_name} docs: {n_docs}')

    doc_inds = np.arange(n_docs)
    # np.random.seed(777)
    np.random.shuffle(doc_inds)
    n_docs_val = int(n_docs * val_ratio)
    n_docs_train = n_docs - n_docs_val
    doc_inds_train, doc_inds_val = doc_inds[:n_docs_train].copy(), doc_inds[n_docs_train:].copy()

    if shuffle:
        np.random.shuffle(doc_inds_train)
        np.random.shuffle(doc_inds_val)

    train_batch_it = EedWikiIterator2(
        ds=ds, inds=doc_inds_train, inp_len=inp_len, tkz=tkz, docs_batch_size=docs_batch_size, device=device,
        preserve_edge_tokens=True, conseq=mask_conseq,
    ).get_batch_iterator()
    val_batch_it = EedWikiIterator2(
        ds=ds, inds=doc_inds_val, inp_len=inp_len, tkz=tkz, docs_batch_size=docs_batch_size, device=device,
        preserve_edge_tokens=True, conseq=mask_conseq,
    ).get_batch_iterator()
    return train_batch_it, val_batch_it


class QnaQuesInp(str, Enum):
    Enc = 'enc'
    Dec = 'dec'


class QnaBatch:
    qas: list[tuple[str, str]]
    contexts: list[str]
    toks_seq_len: int
    tkz: PreTrainedTokenizer
    ques_inp: QnaQuesInp
    q_toks: list[np.ndarray]
    a_toks: list[np.ndarray]
    qa_toks: list[np.ndarray]
    a_att_masks: list[np.ndarray]
    a_tgt_masks: list[np.ndarray]
    qa_att_masks: list[np.ndarray]
    qa_tgt_masks: list[np.ndarray]
    ctx_toks: np.ndarray
    qa_toks_t: list[torch.Tensor]
    q_toks_t: list[torch.Tensor]
    a_toks_t: list[torch.Tensor]
    a_att_masks_t: list[torch.Tensor]
    a_tgt_masks_t: list[torch.Tensor]
    qa_att_mask_t: list[torch.Tensor]
    qa_tgt_mask_t: list[torch.Tensor]
    device: Optional[torch.device] = None
    ctx_toks_t: Optional[torch.Tensor] = None

    def __init__(
            self, qas: list[tuple[str, str]], contexts: list[str], toks_seq_len: int, tkz: PreTrainedTokenizer,
            ques_inp: QnaQuesInp, device: Optional[torch.device] = None,
    ):
        self.qas = qas
        self.contexts = contexts
        self.toks_seq_len = toks_seq_len
        self.tkz = tkz
        self.ques_inp = ques_inp
        self.device = device
        self._process()

    def _process(self):
        # Question + Answer
        q_toks_l, a_toks_l, qa_toks_l, a_att_masks_l, a_tgt_masks_l, qa_att_masks_l, qa_tgt_masks_l = [], [], [], [], [], [], []
        qas_cum, qas_sq_cum, as_cum = 0, 0, 0
        for q, a in self.qas:
            q = f'Question: {q}. Answer: '
            q_toks: list[int] = self.tkz(q).input_ids
            a_toks: list[int] = self.tkz(a).input_ids
            assert q_toks[0] == a_toks[0] == self.tkz.cls_token_id, f'q_toks[0] = {q_toks[0]}. a_toks[0] = {a_toks[0]}'
            assert q_toks[-1] == a_toks[-1] == self.tkz.sep_token_id, f'q_toks[-1] = {q_toks[-1]}. a_toks[-1] = {a_toks[-1]}'
            q_toks, a_toks = q_toks[0:-1], a_toks[1:]

            # # TODO: parametrize
            # if self.ques_inp == QnaQuesInp.Dec:
            #     if len(a_toks) > 18:
            #         a_toks = a_toks[:17] + a_toks[-1:]
            #     if len(a_toks) > 10 and len(q_toks) + len(a_toks) > 30:
            #         if len(q_toks) > 20:
            #             q_toks = q_toks[:1] + q_toks[-19:]
            #         a_toks = a_toks[:9] + a_toks[-1:]
            # elif self.ques_inp == QnaQuesInp.Enc and len(a_toks) > 20:
            #     a_toks = a_toks[:-19] + a_toks[-1:]

            assert len(a_toks) > 1 and a_toks[0] != self.tkz.cls_token_id and a_toks[-1] == self.tkz.sep_token_id, \
                f'a_toks must contain at least one content token and SEP token at the end. a_toks = {a_toks}'
            assert len(q_toks) > 1 and q_toks[0] == self.tkz.cls_token_id and q_toks[-1] != self.tkz.sep_token_id, \
                f'q_toks must contain at least one content token and CLS token at the start. q_toks = {q_toks}'


            qa_toks = [*q_toks, self.tkz.sep_token_id, *a_toks]

            # # TODO: parametrize
            # qa_len = len(qa_toks)
            # qa_len_sq = qa_len**2
            # a_len = len(a_toks)
            # if self.ques_inp == QnaQuesInp.Dec and \
            #         (qas_sq_cum + qa_len_sq >= 2800 or as_cum + a_len > 25 or qas_cum + qa_len > 10000):
            #     continue
            # if self.ques_inp == QnaQuesInp.Enc and as_cum + a_len > 30:
            #     continue
            # qas_cum += qa_len
            # qas_sq_cum += qa_len_sq
            # as_cum += a_len

            if self.ques_inp == QnaQuesInp.Enc:
                assert q_toks[-1] != self.tkz.sep_token_id
                n_toks = len(q_toks)
                if n_toks > self.toks_seq_len:
                    q_toks = q_toks[:self.toks_seq_len]
                elif n_toks < self.toks_seq_len:
                    n_pad = self.toks_seq_len - n_toks
                    q_toks += [self.tkz.pad_token_id] * n_pad

            q_toks_l.append(np.array(q_toks, dtype=int))
            a_toks_l.append(np.array(a_toks, dtype=int))
            qa_toks_l.append(np.array(qa_toks, dtype=int))

            n_q_toks, n_a_toks = len(q_toks), len(a_toks)
            q_mask = np.ones((n_a_toks, n_q_toks + 1), dtype=int)
            a_att_mask = np.ones((n_a_toks, n_a_toks), dtype=int)
            a_att_mask = np.tril(a_att_mask, k=0)
            a_tgt_mask = np.eye(n_a_toks, dtype=int)
            qa_att_mask = np.concatenate([q_mask, a_att_mask], axis=1)
            qa_tgt_mask = np.concatenate([q_mask * 0, a_tgt_mask], axis=1).astype(bool)
            a_att_masks_l.append(a_att_mask)
            a_tgt_masks_l.append(a_tgt_mask.astype(bool))
            qa_att_masks_l.append(qa_att_mask)
            qa_tgt_masks_l.append(qa_tgt_mask)
        self.q_toks = q_toks_l
        self.a_toks = a_toks_l
        self.qa_toks = qa_toks_l
        self.a_att_masks = a_att_masks_l
        self.a_tgt_masks = a_tgt_masks_l
        self.qa_att_masks = qa_att_masks_l
        self.qa_tgt_masks = qa_tgt_masks_l

        # Contexts
        ctxs = []
        max_ctx_chunks = 3
        for ctx in self.contexts:
            ctx_toks = self.tkz(ctx).input_ids
            assert ctx_toks[0] == self.tkz.cls_token_id, f'ctx_token[0] (={ctx_toks[0]}) != cls_token_id (={self.tkz.cls_token_id})'
            assert ctx_toks[-1] == self.tkz.sep_token_id, f'ctx_token[-1] (={ctx_toks[-1]}) != sep_token_id (={self.tkz.sep_token_id})'
            ctx_toks = ctx_toks[:-1]
            n_pad = self.toks_seq_len - len(ctx_toks) % self.toks_seq_len
            assert self.tkz.pad_token_id is not None
            ctx_toks = np.pad(ctx_toks, (0, n_pad), constant_values=self.tkz.pad_token_id)
            ctx_toks = ctx_toks[:self.toks_seq_len][None]
            ctxs.append(ctx_toks[:max_ctx_chunks])
        ctxs_all = np.concatenate(ctxs)
        self.ctx_toks = ctxs_all

        # ctxs_lens = np.array([len(c) for c in ctxs])
        # qas_lens = np.array([len(qa) for qa in self.qa_toks])
        # qs_lens = np.array([len(q) for q in self.q_toks])
        # as_lens = np.array([len(a) for a in self.a_toks])
        # print(f'Contexts: {ctxs_lens}. {ctxs_all.shape}')
        # print(f'QAs: {qas_lens}. {qas_lens.sum()}. {np.square(qas_lens).sum()}')
        # print(f'Qs: {qs_lens}. {qs_lens.sum()}. {np.square(qs_lens).sum()}')
        # print(f'As: {as_lens}. {as_lens.sum()}. {np.square(as_lens).sum()}')

    def _to_tensor_single(self, arr: np.ndarray) -> torch.Tensor:
        res = torch.from_numpy(arr)
        if self.device is not None:
            res = res.to(self.device)
        return res

    def _to_tensor_multi(self, arr: list[np.ndarray]) -> list[torch.Tensor]:
        return [self._to_tensor_single(x) for x in arr]

    def gen_tensors(self) -> tuple[torch.Tensor, tuple[list[torch.Tensor], ...]]:
        if self.ques_inp == QnaQuesInp.Enc:
            if self.ctx_toks_t is None:
                self.q_toks_t = self._to_tensor_multi(self.q_toks)
                self.a_toks_t = self._to_tensor_multi(self.a_toks)
                self.a_att_masks_t = self._to_tensor_multi(self.a_att_masks)
                self.a_tgt_masks_t = self._to_tensor_multi(self.a_tgt_masks)
                self.ctx_toks_t = self._to_tensor_single(self.ctx_toks)
            return self.ctx_toks_t, (self.q_toks_t, self.a_toks_t, self.a_att_masks_t, self.a_tgt_masks_t)

        if self.ques_inp == QnaQuesInp.Dec:
            if self.ctx_toks_t is None:
                self.qa_toks_t = self._to_tensor_multi(self.qa_toks)
                self.qa_att_mask_t = self._to_tensor_multi(self.qa_att_masks)
                self.qa_tgt_mask_t = self._to_tensor_multi(self.qa_tgt_masks)
                self.ctx_toks_t = self._to_tensor_single(self.ctx_toks)
            return self.ctx_toks_t, (self.qa_toks_t, self.qa_att_mask_t, self.qa_tgt_mask_t)

        raise Exception(f'Question input type {self.ques_inp} is not supported')


def get_squadv2_df(exclude_empty_answers: bool = False) -> pd.DataFrame:
    ds_name = 'squad_v2'
    ds_sq = load_dataset(ds_name)
    df_sq = pd.concat([ds_sq['train'].to_pandas(), ds_sq['validation'].to_pandas()], axis=0)
    n_total = len(df_sq)
    df_sq = df_sq.sample(n_total)
    if exclude_empty_answers:
        mask = df_sq['answers'].apply(lambda ans: len(ans['text']) > 0)
        df_sq = df_sq[mask]
        print(f'Remove empty answers from dataset {ds_name}. Size: {n_total} --> {len(df_sq)}')
    return df_sq


def split_df(df: pd.DataFrame, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_total = len(df)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    return df.iloc[:n_train], df.iloc[n_train:]


# df_sq: ['id', 'title', 'context', 'question', 'answers']
def get_squadv2_batch(tkz: PreTrainedTokenizer, df_sq: pd.DataFrame, inds: np.ndarray, inp_len: int, device: torch.device, ques_inp: QnaQuesInp) -> QnaBatch:
    df_b = df_sq.iloc[inds]
    ctxs, ctx_num, qas = {}, 0, set()
    for _, row in df_b.iterrows():
        if not row.context in ctxs:
            ctx_num += 1
            ctxs[row.context] = f'Context{ctx_num}'
        answers = row.answers['text']
        if len(answers) == 0:
            answers = ['-']
        for answer in answers:
            # q = f'{ctxs[row.context]}. Question: {row.question}'
            # q = f'Question: {row.question}. Answer: '
            q = row.question
            qa = q, answer
            qas.add(qa)
    contexts = [f'{val}. {key}' for key, val in ctxs.items()]
    qas = list(qas)
    n_qas, n_batch = len(qas), len(df_b)
    # max_sz = 2
    max_sz = len(contexts)
    if n_qas > max_sz:
        np.random.shuffle(qas)
        qas = qas[:max_sz]
    return QnaBatch(qas=qas, contexts=contexts, toks_seq_len=inp_len, tkz=tkz, device=device, ques_inp=ques_inp)


BatchIt = Generator[QnaBatch, None, None]


def get_squadv2_batch_iterator(
        df_sq: pd.DataFrame, tkz: PreTrainedTokenizer, batch_size: int, inp_len: int, device: torch.device, ques_inp: QnaQuesInp,
) -> BatchIt:
    inds = np.arange(len(df_sq))
    batch_off = 0
    while True:
        batch_inds = inds[batch_off:batch_off + batch_size]
        n_cur = len(batch_inds)
        n_rest = batch_size - n_cur
        if n_rest > 0:
            batch_inds = np.concatenate([batch_inds, inds[:n_rest]])
        sq_batch = get_squadv2_batch(tkz=tkz, df_sq=df_sq, inds=batch_inds, inp_len=inp_len, device=device, ques_inp=ques_inp)
        yield sq_batch
        batch_off += batch_size
        if batch_off >= len(inds):
            batch_off = 0
            np.random.shuffle(inds)


def squadv2_batch_to_tensor_iterator(batch_it: BatchIt) -> ChunkTargetToksGen:
    for batch in batch_it:
        ctx_toks_t, (q_toks_t, a_toks_t, a_att_masks_t, a_tgt_masks_t) = batch.gen_tensors()
        for q_toks, a_toks in zip(q_toks_t, a_toks_t):
            q_toks = q_toks[q_toks != batch.tkz.pad_token_id]
            # print(f'q_toks: {len(q_toks)}. a_toks: {len(a_toks)}')
            if len(q_toks) + len(a_toks) > 50:
                continue
            yield ctx_toks_t, q_toks, a_toks, batch


def get_squadv2_tensor_iterators(
        inp_len: int, batch_size: int, ques_inp: QnaQuesInp, exclude_empty_answers: bool, tkz: PreTrainedTokenizer,
        device: torch.device, val_ratio: float = 0.05,
) -> tuple[ChunkTargetToksGen, ChunkTargetToksGen]:
    df_sq = get_squadv2_df(exclude_empty_answers=exclude_empty_answers)
    df_sq_t, df_sq_v = split_df(df_sq, val_ratio=val_ratio)
    print(f'Squad v2 n_total = {len(df_sq)}. n_train = {len(df_sq_t)}. n_val = {len(df_sq_v)}')

    train_batch_it = get_squadv2_batch_iterator(
        df_sq=df_sq_t, tkz=tkz, batch_size=batch_size, inp_len=inp_len, device=device, ques_inp=ques_inp,
    )
    val_batch_it = get_squadv2_batch_iterator(
        df_sq=df_sq_v, tkz=tkz, batch_size=batch_size, inp_len=inp_len, device=device, ques_inp=ques_inp,
    )
    return squadv2_batch_to_tensor_iterator(train_batch_it), squadv2_batch_to_tensor_iterator(val_batch_it)


def qna_loss(logits: torch.Tensor, tokens: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
    tgt_logits = logits[tgt_mask]
    tgt_probs = torch.softmax(tgt_logits, dim=-1)
    tgt_toks = tokens[tgt_mask][..., None]
    tok_probs = torch.gather(tgt_probs, dim=-1, index=tgt_toks)
    tok_logs = torch.log(tok_probs).reshape(len(tgt_mask))
    if len(tok_logs) > 1:
        loss = -(0.95 * torch.mean(tok_logs[:-1]) + 0.05 * tok_logs[-1])
    else:
        loss = -tok_logs[0]
    return loss


# logits: [n_batch, n_seq, d_model] or [n_seq, d_model]
# tokens: [n_batch, n_seq] or [n_seq]
# tokens[..., -1] = sep_tok_id
def gen_loss(logits: torch.Tensor, tokens: torch.Tensor, sep_token_id: int = 102) -> torch.Tensor:
    assert torch.all(tokens[..., -1] == sep_token_id)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(-1)
    probs = torch.softmax(logits, dim=-1)
    probs = torch.gather(probs, dim=-1, index=tokens)
    logs = torch.log(probs)
    if len(logs) > 1:
        loss = -(0.95 * torch.mean(logs[:-1]) + 0.05 * logs[-1])
    else:
        loss = -logs[0]
    return loss


def get_billsum_df() -> pd.DataFrame:
    ds_name = 'billsum'
    billsum = load_dataset(ds_name)
    df = pd.concat([billsum['train'].to_pandas(), billsum['test'].to_pandas(), billsum['ca_test'].to_pandas()], axis=0)
    return df


@dataclass
class QnaTuple:
    ind: int
    context: str
    question: str
    answer: str


QnaTxtGen = Generator[QnaTuple, None, None]


def get_squadv2_txt_iterator(df_squad: pd.DataFrame) -> QnaTxtGen:
    n = len(df_squad)
    inds = np.arange(n)
    i_off = 0
    while True:
        ind = inds[i_off].item()
        row = df_squad.iloc[ind]
        context, question, answers = row['context'], row['question'], row['answers']['text']
        if len(answers) == 0:
            answers = ['-']

        for answer in answers:
            qna_tuple = QnaTuple(ind=ind, context=context, question=question, answer=answer)
            yield qna_tuple

        i_off += 1
        if i_off == n:
            np.random.shuffle(inds)
            i_off = 0


def get_squadv2_txt_iterators(exclude_empty_answers: bool, val_ratio: float = 0.05) -> tuple[QnaTxtGen, QnaTxtGen]:
    df_sq = get_squadv2_df(exclude_empty_answers=exclude_empty_answers)
    df_sq_t, df_sq_v = split_df(df_sq, val_ratio=val_ratio)
    print(f'Squad v2 n_total = {len(df_sq)}. n_train = {len(df_sq_t)}. n_val = {len(df_sq_v)}')

    train_it = get_squadv2_txt_iterator(
        df_squad=df_sq_t,
    )
    val_it = get_squadv2_txt_iterator(
        df_squad=df_sq_v,
    )
    return train_it, val_it

@dataclass
class SumTuple:
    ind: int
    text: str
    summary: str
    title: str


SumTxtGen = Generator[SumTuple, None, None]


def get_billsum_txt_iterator(df_sum: pd.DataFrame) -> SumTxtGen:
    n = len(df_sum)
    inds = np.arange(n)
    i_off = 0
    while True:
        ind = inds[i_off].item()
        row = df_sum.iloc[ind]
        text, summary, title = row['text'], row['summary'], row['title']

        sum_tuple = SumTuple(ind=ind, text=text, summary=summary, title=title)
        yield sum_tuple

        i_off += 1
        if i_off == n:
            np.random.shuffle(inds)
            i_off = 0


def get_billsum_txt_iterators(val_ratio: float = 0.05) -> tuple[SumTxtGen, SumTxtGen]:
    df_bs = get_billsum_df()
    df_bs = df_bs.sample(n=len(df_bs))
    df_bs_t, df_bs_v = split_df(df_bs, val_ratio=val_ratio)
    print(f'Billsum n_total = {len(df_bs)}. n_train = {len(df_bs_t)}. n_val = {len(df_bs_v)}')

    train_it = get_billsum_txt_iterator(
        df_sum=df_bs_t,
    )
    val_it = get_billsum_txt_iterator(
        df_sum=df_bs_v,
    )
    return train_it, val_it


class WordToks:
    tkz: PreTrainedTokenizer
    s: str
    toks_ids: list[int]
    toks_strs: list[str]
    words_inds_lens: list[tuple[int, int]]
    tags_names: list[str] = ['cite_begin', 'cite_end']
    tags_dict: dict[str, str]
    max_tgt_len_freq: float
    max_tgt_len: int
    off_words_tgt: int
    n_words_tgt: int
    inp_toks: list[int]
    inp_masked_toks: list[int]
    tgt_toks: list[int]
    inp_str: str
    inp_masked_str: str
    tgt_str: str

    def __init__(self, tkz: PreTrainedTokenizer, s: str, max_tgt_len_freq: float = 0, max_tgt_len: int = 0, max_toks: int = 0):
        self.tkz = tkz
        self.s = s
        self.toks_ids = self.tkz(s, add_special_tokens=False).input_ids
        if max_toks > 0:
            self.toks_ids = self.toks_ids[:max_toks]
        self.toks_strs = self.tkz.convert_ids_to_tokens(self.toks_ids)
        self.words_inds_lens = self.calc_inds_lens()
        self.tags_dict = {tname: f'<|{tname}|>' for tname in self.tags_names}
        assert max_tgt_len_freq > 0 or max_tgt_len > 0, \
            f'At least max_tgt_len_freq (={max_tgt_len_freq}) or max_tgt_len (={max_tgt_len}) must be positive.'
        self.max_tgt_len_freq = max_tgt_len_freq
        self.max_tgt_len = max_tgt_len
        self.off_words_tgt, self.n_words_tgt = self.gen_words_inds()
        self.inp_toks, self.inp_masked_toks, self.tgt_toks = self.create_tgt_toks()
        self.inp_str, self.inp_masked_str, self.tgt_str = self.get_tgt_strs()
    
    def calc_inds_lens(self) -> list[tuple[int, int]]:
        res = []
        n_toks_ids, n_toks_strs = len(self.toks_ids), len(self.toks_strs)
        assert n_toks_ids == n_toks_strs, f'n_toks_ids (={n_toks_ids}) must be equal to n_toks_strs (={n_toks_strs})'
        assert n_toks_ids > 0
        assert not self.toks_strs[0].startswith('##'), f'First token cannot start from ##. Tokens: {self.toks_strs}'
        if n_toks_ids == 0:
            return res
        off, len_ = 0, 1
        for i in range(1, n_toks_strs):
            tok_str = self.toks_strs[i]
            if not tok_str.startswith('##'):
                res.append((off, len_))
                off, len_ = i, 1
            else:
                len_ += 1
        res.append((off, len_))
        return res

    def gen_words_inds(self) -> tuple[int, int]:
        n_words = len(self.words_inds_lens)
        if self.max_tgt_len <= 0:
            max_len = int(self.max_tgt_len_freq * n_words)
        elif self.max_tgt_len_freq <= 0:
            max_len = self.max_tgt_len
        else:
            max_len = min(self.max_tgt_len, int(self.max_tgt_len_freq * n_words))
        max_len = min(max_len, int(0.5 * n_words))
        max_len = max(max_len, 1)
        cite_len = np.random.randint(1, max_len + 1)
        n_rest = n_words - cite_len
        assert n_words == 1 or n_rest > 0, f'n_rest (={n_rest}) must be positive when n_words (={n_words}) > 1.'
        off = np.random.randint(n_rest + 1)
        return off, cite_len
    
    def create_tgt_toks(self) -> tuple[list[int], list[int], list[int]]:
        tags_toks = {tname: self.tkz(tval, add_special_tokens=False).input_ids for tname, tval in self.tags_dict.items()}
        # print(len(self.words_inds_lens), self.words_inds_lens)
        # print(self.off_words_tgt, self.n_words_tgt)
        i_tgt_beg = self.words_inds_lens[self.off_words_tgt][0]
        i_tgt_end = sum(self.words_inds_lens[self.off_words_tgt + self.n_words_tgt - 1])
        tgt_pre_toks = self.toks_ids[:i_tgt_beg]
        tgt_toks = self.toks_ids[i_tgt_beg:i_tgt_end]
        tgt_nxt_toks = self.toks_ids[i_tgt_end:]
        tag_beg_toks, tag_end_toks = tags_toks['cite_begin'], tags_toks['cite_end']
        inp_toks = [*tgt_pre_toks, *tag_beg_toks, *tgt_toks, *tag_end_toks, *tgt_nxt_toks]
        tgt_mask_toks = [self.tkz.mask_token_id] * len(tgt_toks)
        inp_masked_toks = [*tgt_pre_toks, *tag_beg_toks, *tgt_mask_toks, *tag_end_toks, *tgt_nxt_toks]
        return inp_toks, inp_masked_toks, tgt_toks

    def get_tgt_strs(self) -> tuple[str, str, str]:
        return self.tkz.decode(self.inp_toks), self.tkz.decode(self.inp_masked_toks), self.tkz.decode(self.tgt_toks)


def run_get_wiki_iterators():
    data_path = Path(os.path.expandvars('$HOME')) / 'data'
    train_it, val_it = get_wiki_iterators(
        data_path=data_path,
    )
    item = next(train_it)
    print('Train:', item)
    item = next(val_it)
    print('Val:', item)


def run_mask_seq():
    pretrained_model_name = 'bert-base-uncased'
    tkz = AutoTokenizer.from_pretrained(pretrained_model_name)
    s = 'Hall Films for Thames Television. It was first shown on ITV during its CITV output on weekday afternoons. Four series were made comprising 65 episodes which aired between 6 September 1988'

    # print(tkz)
    s_masked = mask_random_words(
        s, mask_tok_str=tkz.mask_token, rem_freq=0, rem_prob=0, rem_conseq_freq=0.33,
        rem_conseq_prob=0.2, rem_conseq_max_len=20, rem_conseq_max_times=1)
    print(s)
    print(s_masked)


def run_words_tkz():
    pretrained_model_name = 'bert-base-uncased'
    tkz = AutoTokenizer.from_pretrained(pretrained_model_name)
    s = 'This directory holds the individual version scripts. Users of other migration tools may notice that the files here don’t use ascending integers, and instead use a partial GUID approach.'
    wt = WordToks(
        tkz=tkz, s=s, max_tgt_len_freq=0.5, max_tgt_len=15,
    )
    print(wt.s)
    print(wt.toks_strs)
    print(wt.inp_str)
    print(wt.inp_masked_str)
    print(wt.tgt_str)
    print(tkz(wt.tags_dict['cite_begin'], add_special_tokens=False).input_ids)
    print(tkz(wt.tags_dict['cite_end'], add_special_tokens=False).input_ids)


if __name__ == '__main__':
    # run_get_wiki_iterators()
    # run_mask_seq()
    run_words_tkz()

