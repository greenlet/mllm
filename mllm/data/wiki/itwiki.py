from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset

from transformers import PreTrainedTokenizer

from mllm.train.mask_utils import MaskCfg, mask_random_words_v2


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
    src_toks: np.ndarray
    toks: np.ndarray
    masked_toks: np.ndarray
    mask: Optional[np.ndarray]

    def __init__(
            self, tkz: PreTrainedTokenizer, ind: int, title: str, text: str, max_len: int, mask_cfg: MaskCfg):
        self.tkz = tkz
        self.ind = ind
        self.title = title
        self.text = text
        self.max_len = max_len
        self.mask_cfg = mask_cfg
        self.calc_toks()

    def calc_toks(self):
        src_toks = self.tkz(self.text, add_special_tokens=False).input_ids
        src_toks = np.darray(src_toks)
        toks = src_toks
        if self.max_len > 0:
            toks = toks[:self.max_len]

        masked_toks, mask = mask_random_words_v2(toks, self.tkz, self.mask_cfg)

        self.src_toks = src_toks
        self.toks = toks
        self.masked_toks = masked_toks
        self.mask = mask

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
    toks_t: Optional[torch.Tensor] = None
    masked_toks_t: Optional[torch.Tensor] = None
    mask_t: Optional[torch.Tensor] = None

    def __init__(self, items: list[WikiItem], device: Optional[torch.device] = None):
        self.tkz = items[0].tkz
        self.items = items
        self.device = device if device is not None else torch.device('cpu')

    def calc_all(self):
        n_batch, max_len = len(self.items), self.items[0].max_len
        if max_len == 0:
            max_len = max(len(item.toks) for item in self.items)
        b_toks = np.array((n_batch, max_len), dtype=int)
        b_masked_toks = b_toks.copy()
        b_mask = np.zeros_like(b_toks, dtype=bool)
        for i, item in enumerate(self.items):
            n_toks = len(item.toks)
            b_toks[i, :n_toks] = item.toks
            b_masked_toks[i, :n_toks] = item.masked_toks
            if item.mask is not None:
                b_mask[i, :n_toks] = item.mask

        self.toks = b_toks
        self.masked_toks = b_masked_toks
        self.mask = b_mask
        self.has_mask = self.mask.any().item()

    def get_tensors(self) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.toks_t is None:
            self.toks_t = torch.from_numpy(self.toks).to(self.device)
            self.masked_toks_t = torch.from_numpy(self.masked_toks).to(self.device)
            self.mask_t = torch.from_numpy(self.mask).to(self.device)
        return self.toks_t, self.masked_toks_t, self.mask_t


WikiItemGen = Generator[WikiItem, None, None]
WikiBatchGen = Generator[WikiBatch, None, None]


def get_wiki_item_iterator(
        ds: Dataset, tkz: PreTrainedTokenizer, inds: np.ndarray, n_toks_min: int = 20, n_toks_max: int = 0,
        mask_cfg: Optional[MaskCfg] = None,
) -> WikiItemGen:
    n = len(inds)
    i_off = 0
    while True:
        ind = inds[i_off].item()
        row = ds[ind]

        wiki_item = WikiItem(
            tkz=tkz, ind=ind, title=row['title'], text=row['text'], max_len=n_toks_max, mask_cfg=mask_cfg,
        )
        yield wiki_item

        i_off += 1
        if i_off == n:
            np.random.shuffle(inds)
            i_off = 0


def get_wiki_batch_iterator(
        ds: Dataset, tkz: PreTrainedTokenizer, inds: np.ndarray, batch_size: int, n_toks_min: int = 20, n_toks_max: int = 0,
        mask_cfg: Optional[MaskCfg] = None,
    ) -> WikiBatchGen:
    wiki_it = get_wiki_item_iterator(ds, tkz, inds)
    items = []
    for wiki_item in wiki_it:
        if wiki_item.src_toks_num < n_toks_min:
            continue
        items.append(wiki_item)
        if len(items) == batch_size:
            batch = WikiBatch(items=items)
            yield batch


def get_wiki_batch_iterators(
        data_path: Path, tkz: PreTrainedTokenizer, batch_size: int, val_ratio: float = 0.05, shuffle: bool = False, rand_seed: int = 100,
        n_toks_min: int = 20, n_toks_max: int = 0, mask_cfg: Optional[MaskCfg] = None,
) -> tuple[WikiBatchGen, WikiBatchGen]:
    ds, doc_inds_train, doc_inds_val = get_split_wiki_ds(data_path, val_ratio, shuffle, rand_seed)
    train_it = get_wiki_batch_iterator(
        ds=ds, tkz=tkz, inds=doc_inds_train, batch_size=batch_size, n_toks_min=n_toks_min, n_toks_max=n_toks_max, mask_cfg=mask_cfg,
    )
    val_it = get_wiki_batch_iterator(
        ds=ds, tkz=tkz, inds=doc_inds_val, batch_size=batch_size, n_toks_min=n_toks_min, n_toks_max=n_toks_max, mask_cfg=mask_cfg,
    )
    return train_it, val_it

