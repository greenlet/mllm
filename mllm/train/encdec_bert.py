import argparse
from collections import defaultdict
import os
from pathlib import Path
import shutil
from typing import Dict, Generator, List, Tuple, Any, Callable, Optional, Union

from datasets import Dataset, load_dataset
import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import trange
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from mllm.model.bert.modeling_bert import BertModel
from mllm.model.losses import EncdecMaskPadItemLoss
from mllm.train.mask_utils import MaskCfg, mask_random_words_v2



class MaskedDataset(Dataset):
    def __init__(self, dataset: Dataset, tkz: PreTrainedTokenizer, max_seq_len: int, mask_cfg: Optional[MaskCfg] = None):
        self.dataset = dataset
        self.tkz = tkz
        self.size = len(self.dataset)
        self.pad_token_id = tkz.pad_token_id
        self.max_seq_len = max_seq_len
        self.mask_token_id = tkz.mask_token_id
        self.mask_cfg = mask_cfg
        self.inds = np.arange(self.size)

    def __len__(self):
        return self.size

    def extract_masked_input(self, item: Dict) -> Dict[str, Any]:
        toks = self.tkz(item['text'], add_special_tokens=True).input_ids
        cur_len = len(toks)
        max_seq_len = min(cur_len, self.max_seq_len)
        input_ids = toks
        if max_seq_len < cur_len:
            ind_off_max = cur_len - max_seq_len + 1
            ind_off = np.random.randint(0, ind_off_max)
            input_ids = input_ids[ind_off:ind_off + max_seq_len]
        
        # Use mask_cfg if provided, otherwise use original ids
        if self.mask_cfg is not None:
            input_ids_masked, _ = mask_random_words_v2(np.array(input_ids), self.tkz, self.mask_cfg)
            input_ids_masked = input_ids_masked.tolist()
        else:
            input_ids_masked = input_ids
        
        n = len(input_ids)
        assert n == len(input_ids_masked), f'Size mismatch: {n} vs {len(input_ids_masked)}'
 
        return {
            **item,
            'input_ids': input_ids,
            'input_ids_masked': input_ids_masked,
        }

    def _get_item(self, ind: int) -> dict[str, Any]:
        ind = self.inds[ind % self.size]
        ind = ind.item()
        item = self.dataset[ind]
        item = self.extract_masked_input(item)
        return item

    def __getitem__(self, idx: Union[int, List[int]]) -> Dict[str, List[Any]]:
        if isinstance(idx, int):
            idx = [idx]
        
        batch = {
            'input_ids': [],
            'input_ids_masked': [],
        }
        for i in idx:
            item = self._get_item(i)
            input_ids, input_ids_masked = item['input_ids'], item['input_ids_masked']
            n = len(input_ids)
            if n < self.max_seq_len:
                pad_len = self.max_seq_len - n
                input_ids = input_ids + [self.pad_token_id] * pad_len
                input_ids_masked = input_ids_masked + [self.pad_token_id] * pad_len
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_ids_masked = torch.tensor(input_ids_masked, dtype=torch.long)
            batch['input_ids'].append(input_ids)
            batch['input_ids_masked'].append(input_ids_masked)
        return batch

    def shuffle(self, seed: Optional[int] = None) -> 'MaskedDataset':
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.inds)
        else:
            np.random.shuffle(self.inds)
        return self
    

def load_masked_wiki_dataset(
        data_path: Path, tkz: PreTrainedTokenizer, max_seq_len: int, val_split_ratio: float,
        mask_cfg: Optional[MaskCfg] = None, random_seed: Optional[int] = 55,
    ) -> Tuple[Dataset, Dataset]:
    wiki_ds_name, wiki_ds_subdir = '20220301.en', 'wikipedia'
    dataset = load_dataset(wiki_ds_subdir, wiki_ds_name, cache_dir=str(data_path), trust_remote_code=True)['train']

    if random_seed is not None:
        dataset = dataset.shuffle(seed=random_seed)
    
    n_total = len(dataset)
    n_val = int(n_total * val_split_ratio)
    n_train = n_total - n_val
    ds_train = dataset.select(range(n_train))
    ds_val = dataset.select(range(n_train, n_train + n_val))

    ds_train = MaskedDataset(ds_train, tkz, max_seq_len=max_seq_len, mask_cfg=mask_cfg)
    ds_val = MaskedDataset(ds_val, tkz, max_seq_len=max_seq_len, mask_cfg=mask_cfg)
    
    train_sz_str = f'!={n_train}' if len(ds_train) != n_train else ''
    val_sz_str = f'!={n_val}' if len(ds_val) != n_val else ''
    print(f'Loaded masked Wiki dataset. Total size: {len(dataset)}. Train: {len(ds_train)}{train_sz_str}. Val: {len(ds_val)}{val_sz_str}.')
    
    return ds_train, ds_val


def create_dataloader(
        dataset: Dataset, batch_size: int, num_workers: int, distributed: bool, drop_last: bool = False,
    ) -> DataLoader:
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2,
        drop_last=drop_last,
        sampler=sampler,
        pin_memory=True,
    )
    return dataloader


def create_dataloader_iter(
        dataset: Dataset, batch_size: int, num_workers: int, distributed: bool, drop_last: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
    while True:
        rank = dist.get_rank() if distributed else 0
        print(f'R{rank}. Create dataloader. batch_size={batch_size}. num_workers={num_workers}. distributed={distributed}. drop_last={drop_last}.')
        dataloader = create_dataloader(
            dataset, batch_size=batch_size, num_workers=num_workers, distributed=distributed, drop_last=drop_last,
        )
        n = 0
        for batch in dataloader:
            yield batch
            n += 1
        print(f'R{rank}. Dataloader cycle finished. Processed {n} batches. Shuffling dataset for next cycle.')
        dataset.shuffle()


