from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Union

import numpy as np
import torch
from datasets import Dataset, load_dataset

from transformers import PreTrainedTokenizer

from mllm.config.model import SelfSuperviseType
from mllm.train.mask_utils import MaskCfg, mask_random_words_v2
from nltk.tokenize import sent_tokenize


def get_split_wiki_ds(data_path: Path, val_ratio: float = 0.05, shuffle: bool = False, rand_seed: Optional[int] = None) -> tuple[Dataset, np.ndarray, np.ndarray]:
    # wiki_ds_name, wiki_ds_subdir = '20200501.en', 'wikipedia'
    # dss = load_dataset(wiki_ds_subdir, wiki_ds_name, beam_runner='DirectRunner', cache_dir=str(data_path))
    wiki_ds_name, wiki_ds_subdir = '20220301.en', 'wikipedia'
    dss = load_dataset(wiki_ds_subdir, wiki_ds_name, cache_dir=str(data_path))
    ds = dss['train']
    n_docs = len(ds)
    print(f'Wikipedia {wiki_ds_name} docs: {n_docs}')

    doc_inds = np.arange(n_docs)
    if rand_seed is not None:
        np.random.seed(rand_seed)
    np.random.shuffle(doc_inds)
    n_docs_val = int(n_docs * val_ratio)
    n_docs_train = n_docs - n_docs_val
    doc_inds_train, doc_inds_val = doc_inds[:n_docs_train].copy(), doc_inds[n_docs_train:].copy()

    if shuffle:
        np.random.shuffle(doc_inds_train)
        np.random.shuffle(doc_inds_val)

    return ds, doc_inds_train, doc_inds_val


@dataclass
class WikiTuple:
    int: int
    title: str
    text: str


WikiTupleGen = Generator[WikiTuple, None, None]


def get_wiki_iterator(ds: Dataset, inds: np.ndarray) -> WikiTupleGen:
    n = len(inds)
    i_off = 0
    while True:
        ind = inds[i_off].item()
        row = ds[ind]

        wiki_item = WikiTuple(ind=ind, title=row['title'], text=row['text'])
        yield wiki_item

        i_off += 1
        if i_off == n:
            np.random.shuffle(inds)
            i_off = 0


def get_wiki_iterators(data_path: Path, val_ratio: float = 0.05, shuffle: bool = False, rand_seed: int = 100) -> tuple[WikiTupleGen, WikiTupleGen]:
    ds, doc_inds_train, doc_inds_val = get_split_wiki_ds(data_path, val_ratio, shuffle, rand_seed)
    train_it = get_wiki_iterator(ds, doc_inds_train)
    val_it = get_wiki_iterator(ds, doc_inds_val)
    return train_it, val_it



class WikiItem:
    tkz: PreTrainedTokenizer
    ind: int
    title: str
    text: str
    max_len: int
    mask_cfg: Optional[MaskCfg]
    self_supervise_type: SelfSuperviseType
    max_pred_len: int
    src_toks: np.ndarray
    toks: np.ndarray
    masked_toks: np.ndarray
    mask: Optional[np.ndarray]
    tgt_toks: Optional[np.ndarray]

    def __init__(
            self, tkz: PreTrainedTokenizer, ind: int, title: str, text: str, max_len: int, mask_cfg: Optional[MaskCfg],
            self_supervise_type: SelfSuperviseType, max_pred_len: int,
    ):
        self.tkz = tkz
        self.ind = ind
        self.title = title
        self.text = text
        self.max_len = max_len
        self.mask_cfg = mask_cfg
        self.self_supervise_type = self_supervise_type
        self.max_pred_len = max_pred_len
        self.calc_toks()

    def _tokenize(self, s: str, to_numpy: bool = True) -> Union[list, np.ndarray]:
        toks = self.tkz(s, add_special_tokens=False).input_ids
        if to_numpy:
            toks = np.array(toks)
        return toks

    def calc_toks_sents(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        min_sents_num = 2
        min_sents_rand = 10
        min_toks_ratio = 0.8
        min_toks_total = int((self.max_len + self.max_pred_len) * min_toks_ratio)
        sents = sent_tokenize(self.text, language='english')
        n_sents = len(sents)
        if n_sents < min_sents_num:
            toks = self._tokenize(self.text)
            n_toks = len(toks)
            if n_toks < min_toks_total:
                return toks[:self.max_len], None
            tgt_ratio = self.max_pred_len / (self.max_len + self.max_pred_len)
            n_tgt = int(n_toks * tgt_ratio)
            n_inp = n_toks - n_tgt
            toks_inp, toks_tgt = toks[:n_inp], toks[n_inp:]
            if len(toks_inp) > self.max_len:
                toks_inp = toks_inp[-self.max_len:]
            if len(toks_tgt) > self.max_pred_len:
                toks_tgt = toks_tgt[:self.max_pred_len]
            return toks_inp, toks_tgt

        if n_sents >= min_sents_rand:
            i = np.random.randint(n_sents - min_sents_rand + 1)
            sents = sents[i:]
            n_sents = len(sents)

        n_inp_toks, n_tgt_toks = 0, 0
        sents_inp, sents_tgt = [], []
        for i, sent in enumerate(sents):
            sent_toks = self._tokenize(sent, to_numpy=False)
            n_sent_toks = len(sent_toks)
            if n_inp_toks < self.max_len and i < len(sents) - 1:
                sents_inp.append(sent)
                n_inp_toks += n_sent_toks
            else:
                sents_tgt.append(sent)
                n_tgt_toks += n_sent_toks
                if n_tgt_toks >= self.max_pred_len:
                    break

        txt_inp, txt_tgt = ' '.join(sents_inp), ' '.join(sents_tgt)
        toks_inp, toks_tgt = self._tokenize(txt_inp), self._tokenize(txt_tgt)
        assert len(toks_inp) > 0
        assert len(toks_tgt) > 0
        if len(toks_inp) > self.max_len:
            toks_inp = toks_inp[-self.max_len:]
        if len(toks_tgt) > self.max_pred_len:
            toks_tgt = toks_tgt[:self.max_pred_len]
        return toks_inp, toks_tgt

    def calc_toks(self):
        if self.self_supervise_type == SelfSuperviseType.Input:
            src_toks = self._tokenize(self.text)
            toks = src_toks
            if self.max_len > 0:
                toks = toks[:self.max_len]
            tgt_toks = None
        elif self.self_supervise_type == SelfSuperviseType.NextSent:
            toks, tgt_toks = self.calc_toks_sents()
            src_toks = toks
        elif self.self_supervise_type == SelfSuperviseType.NextTok:
            src_toks = self._tokenize(self.text)
            n_src_toks = len(src_toks)
            if n_src_toks >= self.max_len + self.max_pred_len:
                max_len, max_pred_len = self.max_len, self.max_pred_len
            elif n_src_toks >= self.max_len // 2 + self.max_pred_len // 2:
                max_len, max_pred_len = self.max_len // 2, self.max_pred_len // 2
            else:
                max_len, max_pred_len = 0, 0
            if max_len > 0:
                n1 = np.random.randint(2, max_len + 1)
                n2 = np.random.randint(1, min(n1 // 2, max_pred_len) + 1)
                n_total = n1 + n2
                n_rest = n_src_toks - n_total
                i_off = np.random.randint(n_rest + 1)
                toks = src_toks[i_off:i_off + n1]
                tgt_toks = src_toks[i_off + n1:i_off + n1 + n2]
            else:
                toks = src_toks
                tgt_toks = None
        else:
            raise ValueError(f'Self supervised type {self.self_supervise_type} is not supported.')

        masked_toks, mask = toks, None
        if self.mask_cfg is not None:
            masked_toks, mask = mask_random_words_v2(toks, self.tkz, self.mask_cfg)

        self.src_toks = src_toks
        self.toks = toks
        self.masked_toks = masked_toks
        self.mask = mask
        self.tgt_toks = tgt_toks

    @property
    def src_toks_num(self) -> int:
        return len(self.src_toks)


class WikiBatch:
    tkz: PreTrainedTokenizer
    items: list[WikiItem]
    device: torch.device
    toks: np.ndarray
    masked_toks: np.ndarray
    mask: np.ndarray
    has_mask: bool
    tgt_toks: Optional[np.ndarray] = None
    toks_t: Optional[torch.Tensor] = None
    masked_toks_t: Optional[torch.Tensor] = None
    mask_t: Optional[torch.Tensor] = None
    tgt_toks_t: Optional[torch.Tensor] = None

    def __init__(self, items: list[WikiItem], device: Optional[torch.device] = None):
        self.tkz = items[0].tkz
        self.items = items
        self.device = device if device is not None else torch.device('cpu')
        self.calc_all()

    def calc_all(self):
        n_batch, max_len = len(self.items), self.items[0].max_len
        next_sent_pred = self.items[0].tgt_toks is not None
        if max_len == 0 or next_sent_pred:
            max_len = max(len(item.toks) for item in self.items)
        b_toks = np.full((n_batch, max_len), self.tkz.pad_token_id, dtype=int)
        b_masked_toks = b_toks.copy()
        b_mask = np.zeros_like(b_toks, dtype=bool)
        for i, item in enumerate(self.items):
            n_toks = len(item.toks)
            b_toks[i, :n_toks] = item.toks
            b_masked_toks[i, :n_toks] = item.masked_toks
            if item.mask is not None:
                b_mask[i, :n_toks] = item.mask

        b_tgt_toks = None
        if next_sent_pred:
            max_tgt_len = max(len(item.tgt_toks) for item in self.items)
            b_tgt_toks = np.full((n_batch, max_tgt_len), self.tkz.pad_token_id, dtype=int)
            for i, item in enumerate(self.items):
                n_tgt_toks = len(item.tgt_toks)
                b_tgt_toks[i, :n_tgt_toks] = item.tgt_toks

        self.toks = b_toks
        self.masked_toks = b_masked_toks
        self.mask = b_mask
        self.has_mask = self.mask.any().item()
        self.tgt_toks = b_tgt_toks

    def get_tensors(self) -> [torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.toks_t is None:
            self.toks_t = torch.from_numpy(self.toks).to(self.device)
            self.masked_toks_t = torch.from_numpy(self.masked_toks).to(self.device)
            self.mask_t = torch.from_numpy(self.mask).to(self.device)
            if self.tgt_toks is not None:
                self.tgt_toks_t = torch.from_numpy(self.tgt_toks).to(self.device)
        return self.toks_t, self.masked_toks_t, self.mask_t, self.tgt_toks_t


WikiItemGen = Generator[WikiItem, None, None]
WikiBatchGen = Generator[WikiBatch, None, None]


def get_wiki_item_iterator(
        ds: Dataset, tkz: PreTrainedTokenizer, inds: np.ndarray, n_toks_max: int = 0, mask_cfg: Optional[MaskCfg] = None,
        self_supervise_type: SelfSuperviseType = SelfSuperviseType.Input, n_toks_pred_max: int = 0,
) -> WikiItemGen:
    n = len(inds)
    i_off = 0
    while True:
        ind = inds[i_off].item()
        row = ds[ind]

        wiki_item = WikiItem(
            tkz=tkz, ind=ind, title=row['title'], text=row['text'], max_len=n_toks_max, mask_cfg=mask_cfg, self_supervise_type=self_supervise_type,
            max_pred_len=n_toks_pred_max,
        )
        yield wiki_item

        i_off += 1
        if i_off == n:
            np.random.shuffle(inds)
            i_off = 0


def get_wiki_batch_iterator(
        ds: Dataset, tkz: PreTrainedTokenizer, inds: np.ndarray, batch_size: int, n_toks_min: int = 20, n_toks_max: int = 0,
        mask_cfg: Optional[MaskCfg] = None, device: Optional[torch.device] = None, self_supervise_type: SelfSuperviseType = SelfSuperviseType.Input,
        n_toks_pred_max: int = 0,
    ) -> WikiBatchGen:
    wiki_it = get_wiki_item_iterator(
        ds=ds, tkz=tkz, inds=inds, n_toks_max=n_toks_max, mask_cfg=mask_cfg, self_supervise_type=self_supervise_type, n_toks_pred_max=n_toks_pred_max,
    )
    items = []
    for wiki_item in wiki_it:
        if wiki_item.src_toks_num < n_toks_min:
            continue
        if self_supervise_type in (SelfSuperviseType.NextSent, SelfSuperviseType.NextTok) and wiki_item.tgt_toks is None:
            continue
        items.append(wiki_item)
        if len(items) == batch_size:
            batch = WikiBatch(items=items, device=device)
            yield batch
            items = []


def get_wiki_batch_iterators(
        data_path: Path, tkz: PreTrainedTokenizer, batch_size: int, val_ratio: float = 0.05, shuffle: bool = False, rand_seed: Optional[int] = None,
        n_toks_min: int = 20, n_toks_max: int = 0, mask_cfg: Optional[MaskCfg] = None, device: Optional[torch.device] = None,
        self_supervise_type: SelfSuperviseType = SelfSuperviseType.Input, n_toks_pred_max: int = 0,
) -> tuple[WikiBatchGen, WikiBatchGen]:
    ds, doc_inds_train, doc_inds_val = get_split_wiki_ds(
        data_path=data_path, val_ratio=val_ratio, shuffle=shuffle, rand_seed=rand_seed
    )
    train_it = get_wiki_batch_iterator(
        ds=ds, tkz=tkz, inds=doc_inds_train, batch_size=batch_size, n_toks_min=n_toks_min, n_toks_max=n_toks_max, mask_cfg=mask_cfg,
        device=device, self_supervise_type=self_supervise_type, n_toks_pred_max=n_toks_pred_max,
    )
    val_it = get_wiki_batch_iterator(
        ds=ds, tkz=tkz, inds=doc_inds_val, batch_size=batch_size, n_toks_min=n_toks_min, n_toks_max=n_toks_max, mask_cfg=mask_cfg,
        device=device, self_supervise_type=self_supervise_type, n_toks_pred_max=n_toks_pred_max,
    )
    return train_it, val_it

