import argparse
from collections import defaultdict
from dataclasses import dataclass
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

from mllm.data.utils import RandomInputTokenizerV2, TokensSubsetV2
from mllm.model.bert.modeling_bert import BertModel
from mllm.model.losses import EncdecMaskPadItemLoss
from mllm.train.mask_utils import MaskCfg, mask_random_words_v2


@dataclass(kw_only=True)
class MaskedCiteBatch:
    tokens_subsets: List[TokensSubsetV2]
    # (batch_size, seq_len)
    inp_toks: torch.Tensor
    # (batch_size, seq_len)
    prompts_toks: torch.Tensor
    # (batch_size, seq_len)
    cites_masked_toks: torch.Tensor
    # (batch_size, seq_len)
    cites_toks: torch.Tensor
    # (batch_size, seq_len)
    inp_att_mask: torch.Tensor
    # (batch_size, seq_len)
    prompts_att_mask: torch.Tensor
    # (batch_size, seq_len)
    cites_masked_att_mask: torch.Tensor
    # (batch_size, seq_len)
    cites_att_mask: torch.Tensor
    # (2, batch_size + 1)
    edge_inds: torch.Tensor


class MaskedCiteDataset:
    def __init__(
            self, dataset: Dataset, tkz: PreTrainedTokenizer, max_seq_len: int, n_special_toks: int = 1000, mask_cfg: Optional[MaskCfg] = None,
            device: Optional[torch.device] = None,
        ):
        '''Dataset for masked citation prediction with random input tokens.
         Args:
            dataset: HuggingFace dataset with 'text' field.
            tkz: PreTrainedTokenizer for tokenization.
            max_seq_len: Maximum sequence length.
            n_special_toks: Number of special tokens reserved in tokenizer. 1000 is the number of first BERT token ids reserved for special or unused tokens.
            mask_cfg: Optional MaskCfg for citation masking.
            device: Optional torch.device to move batches to.
        '''
        self.dataset = dataset
        self.tkz = tkz
        self.size = len(self.dataset)
        self.pad_token_id = tkz.pad_token_id
        self.max_seq_len = max_seq_len
        self.mask_token_id = tkz.mask_token_id
        self.mask_cfg = mask_cfg
        self.device = device if device is not None else torch.device('cpu')
        self.inds = np.arange(self.size)
        self.random_inp_tkz = RandomInputTokenizerV2(
            tkz, max_len=max_seq_len, n_random_toks=3, n_special_toks=n_special_toks, mask_cfg=mask_cfg,
        )

    def __len__(self):
        return self.size

    def toks_to_tensor(self, toks_list: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(toks_list)
        max_len = max(len(toks) for toks in toks_list)
        toks_tensor = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long, device=self.device)
        att_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        for i in range(batch_size):
            cur_len = len(toks_list[i])
            toks_tensor[i, :cur_len] = torch.tensor(toks_list[i], dtype=torch.long, device=self.device)
            att_mask[i, :cur_len] = 1
        return toks_tensor, att_mask

    def get_batch(self, inds: List[int]) -> MaskedCiteBatch:
        texts = []
        for i in inds:
            item = self.dataset[i]
            texts.append(item['text'])
        tokens_subsets = self.random_inp_tkz(texts)

        batch_size = len(inds)
        inp_toks, inp_att_mask = self.toks_to_tensor([item.toks_inp for item in tokens_subsets])
        prompts_toks, prompts_att_mask = self.toks_to_tensor([item.toks_prompt for item in tokens_subsets])
        cites_masked_toks, cites_masked_att_mask = self.toks_to_tensor([item.toks_cite_masked for item in tokens_subsets])
        cites_toks, cites_att_mask = self.toks_to_tensor([item.toks_cite for item in tokens_subsets])

        edge_inds = torch.stack([
            torch.arange(batch_size, device=self.device),
            torch.full((batch_size,), batch_size, device=self.device),
        ])

        return MaskedCiteBatch(
            tokens_subsets=tokens_subsets,
            inp_toks=inp_toks,
            prompts_toks=prompts_toks,
            cites_masked_toks=cites_masked_toks,
            cites_toks=cites_toks,
            inp_att_mask=inp_att_mask,
            prompts_att_mask=prompts_att_mask,
            cites_masked_att_mask=cites_masked_att_mask,
            cites_att_mask=cites_att_mask,
            edge_inds=edge_inds,
        )

    def shuffle(self, seed: Optional[int] = None) -> 'MaskedDataset':
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.inds)
        else:
            np.random.shuffle(self.inds)
        return self
    

def load_split_wiki_dataset(
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

    print(f'Loaded masked Wiki dataset. Total size: {len(dataset)}. Train: {len(ds_train)}. Val: {len(ds_val)}.')
    
    return ds_train, ds_val


def create_masked_cite_dataloader(
        dataset: MaskedCiteDataset, batch_size: int, shuffle: bool = True,
    ) -> Generator[MaskedCiteBatch, None, None]:
    '''Create DataLoader for MaskedCiteDataset.
     Args:
        dataset: MaskedCiteDataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data when reaching the end of the dataset.
        Returns:
            Generator yielding MaskedCiteBatch instances.
    '''
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f'R{rank}. Create MaskedCiteDataset dataloader. batch_size={batch_size}. shuffle={shuffle}.')
    start_ind = 0
    while True:
        end_ind = min(start_ind + batch_size, len(dataset))
        inds = dataset.inds[start_ind:end_ind].tolist()
        if len(inds) < batch_size:
            inds += dataset.inds[:(batch_size - len(inds))].tolist()
            print(f'R{rank}. Shuffle dataset')
            dataset.shuffle()
        batch = dataset.get_batch(inds)
        yield batch
        start_ind = end_ind % len(dataset)

