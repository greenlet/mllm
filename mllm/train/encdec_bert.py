import argparse
import os
from pathlib import Path
import shutil
from typing import Dict, List, Tuple, Any, Callable, Optional, Union

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
from mllm.train.mask_utils import MaskCfg



class MaskedDataset(Dataset):
    def __init__(self, dataset: Dataset, tkz: PreTrainedTokenizer, max_seq_len: int, min_mask_toks: int, max_mask_toks: int):
        self.dataset = dataset.map(tokenize_item, fn_kwargs={'tokenizer': tkz})
        self.len = len(dataset)
        self.pad_token_id = tkz.pad_token_id
        self.max_seq_len = max_seq_len
        self.mask_token_id = tkz.mask_token_id
        self.min_mask_toks = min_mask_toks
        self.max_mask_toks = max_mask_toks
        self.inds = np.arange(self.max_seq_len)

    def __len__(self):
        return self.len * 1000

    def extract_masked_input(self, item: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        cur_len = item['toks_len']
        max_seq_len = min(cur_len, self.max_seq_len)
        input_ids = item['toks']
        if max_seq_len < cur_len:
            ind_off_max = cur_len - max_seq_len + 1
            ind_off = np.random.randint(0, ind_off_max)
            input_ids = input_ids[ind_off:ind_off + max_seq_len]
        input_ids_masked = input_ids
        mask_toks_num = np.random.randint(self.min_mask_toks, self.max_mask_toks + 1)
        mask_toks_num = min(mask_toks_num, max_seq_len // 2)
        if mask_toks_num > 0:
            mask_inds = np.random.choice(self.inds[:max_seq_len], size=mask_toks_num, replace=False)
            input_ids_masked = np.array(input_ids)
            input_ids_masked[mask_inds] = self.mask_token_id
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(input_ids_masked, dtype=torch.long)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        idx = idx % len(self.dataset)
        item = self.dataset[idx]
        input_ids, input_ids_masked = self.extract_masked_input(item)
        return {
            **item,
            'input_ids': input_ids,
            'input_ids_masked': input_ids_masked,
        }
    


def load_masked_wiki_dataset(
        data_path: Path, tkz: PreTrainedTokenizer, max_seq_len: int, val_split_ratio: float,
        mask_cfg: Optional[MaskCfg] = None, random_seed: Optional[int] = 55,
    ) -> Tuple[PreTrainedTokenizer, Dataset, Dataset]:
    wiki_ds_name, wiki_ds_subdir = '20220301.en', 'wikipedia'
    dataset = load_dataset(wiki_ds_subdir, wiki_ds_name, cache_dir=str(data_path))[wiki_ds_subdir]['train']

    if random_seed is not None:
        dataset = dataset.shuffle(seed=random_seed)
    
    n_total = len(dataset)
    n_val = int(n_total * val_split_ratio)
    n_train = n_total - n_val
    ds_train = dataset[:n_train]
    ds_val = dataset[n_train:]

    dataset = MaskedDataset(dataset, tkz, max_seq_len=max_seq_len, min_mask_toks=min_mask_toks, max_mask_toks=max_mask_toks)
    return tkz, ds_train, ds_val


