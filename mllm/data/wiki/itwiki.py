from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import numpy as np
from datasets import Dataset, load_dataset

from transformers import PreTrainedTokenizer

from mllm.train.mask_utils import MaskCfg


def get_split_wiki_ds(data_path: Path, val_ratio: float = 0.05, shuffle: bool = False, rand_seed: int = 100) -> tuple[Dataset, np.ndarray, np.ndarray]:
    # wiki_ds_name, wiki_ds_subdir = '20200501.en', 'wikipedia'
    # dss = load_dataset(wiki_ds_subdir, wiki_ds_name, beam_runner='DirectRunner', cache_dir=str(data_path))
    wiki_ds_name, wiki_ds_subdir = '20220301.en', 'wikipedia'
    dss = load_dataset(wiki_ds_subdir, wiki_ds_name, cache_dir=str(data_path))
    ds = dss['train']
    n_docs = len(ds)
    print(f'Wikipedia {wiki_ds_name} docs: {n_docs}')

    doc_inds = np.arange(n_docs)
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
    mask_cfg: MaskCfg
    src_toks: list[int]
    toks: list[int]
    masked_toks: Optional[list[int]]
    mask: Optional[list[bool]]

    def __init__(
            self, tkz: PreTrainedTokenizer, ind: int, title: str, text: str, max_len: int = 0, mask_min_len: int = 0,
            mask_max_len: int = 0, mask_min_rate: float = 0, mask_max_rate: float = 0):
        self.tkz = tkz
        self.ind = ind
        self.title = title
        self.text = text
        self.max_len = max_len
        assert mask_min_len > mask_max_len or 0 <= mask_min_len, f'mask_min_len = {mask_min_len}. mask_max_len = {mask_max_len}'
        assert mask_min_rate > mask_max_rate or 0 <= mask_min_rate <= mask_max_rate <= 1, f'mask_min_rate = {mask_min_rate}. mask_max_rate = {mask_max_rate}'
        self.mask_min_len = mask_min_len
        self.mask_max_len = mask_max_len
        self.mask_min_rate = mask_min_rate
        self.mask_max_rate = mask_max_rate
        self.calc_toks()

    def calc_toks(self):
        src_toks = self.tkz(self.text, add_special_tokens=False).input_ids
        toks = src_toks
        if self.max_len > 0:
            toks = toks[:self.max_len]

        n_toks = len(toks)
        if 0 < self.mask_min_len <= self.mask_max_len:
            mask_min_len = min(self.mask_min_len, n_toks)
            mask_max_len = min(self.mask_max_len, n_toks)
            if mask_min_len == mask_max_len:
                mask_len = mask_min_len
            else:
                mask_len = ...


        self.src_toks = src_toks
        self.toks = toks

    @property
    def src_toks_num(self) -> int:
        return len(self.src_toks)


class WikiBatch:
    tkz: PreTrainedTokenizer
    items: list[WikiItem]

    def __init__(self, items: list[WikiItem]):
        self.tkz = items[0].tkz
        self.items = items


WikiItemGen = Generator[WikiItem, None, None]
WikiBatchGen = Generator[WikiBatch, None, None]


def get_wiki_item_iterator(ds: Dataset, tkz: PreTrainedTokenizer, inds: np.ndarray) -> WikiItemGen:
    n = len(inds)
    i_off = 0
    while True:
        ind = inds[i_off].item()
        row = ds[ind]

        wiki_item = WikiItem(tkz=tkz, ind=ind, title=row['title'], text=row['text'])
        yield wiki_item

        i_off += 1
        if i_off == n:
            np.random.shuffle(inds)
            i_off = 0


def get_wiki_batch_iterator(ds: Dataset, tkz: PreTrainedTokenizer, inds: np.ndarray, batch_size: int, n_toks_min: int = 20) -> WikiBatchGen:
    wiki_it = get_wiki_item_iterator(ds, tkz, inds)
    items = []
    for wiki_item in wiki_it:
        if wiki_item.src_toks_num < n_toks_min:
            continue
        items.append(wiki_item)
        if len(items) == batch_size:
            batch = WikiBatch(items=items)
            yield batch


def get_wiki_batch_iterators(data_path: Path, val_ratio: float = 0.05, shuffle: bool = False, rand_seed: int = 100, n_words_min: int = 20) -> tuple[WikiBatchGen, WikiBatchGen]:
    ds, doc_inds_train, doc_inds_val = get_split_wiki_ds(data_path, val_ratio, shuffle, rand_seed)
    train_it = get_wiki_batch_iterator(ds, doc_inds_train, n_toks_min=n_words_min)
    val_it = get_wiki_batch_iterator(ds, doc_inds_val, n_toks_min=n_words_min)
    return train_it, val_it
