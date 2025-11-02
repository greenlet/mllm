import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

from datasets import Dataset, load_dataset
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from mllm.model.bert.modeling_bert import BertModel


class MaskedBert(nn.Module):
    def __init__(self, share_inout_embeddings: bool = True):
        super(MaskedBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.share_inout_embeddings = share_inout_embeddings
        if not share_inout_embeddings:
            self.output_linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)

    # x: (batch_size, seq_length) --> (batch_size, seq_length, vocab_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(x)
        # y: (batch_size, seq_length, hidden_size)
        y = out.last_hidden_state
        if not self.share_inout_embeddings:
            # y: (batch_size, seq_length, vocab_size)
            y = self.output_linear(y)
        else:
            # y: (batch_size, vocab_size, seq_length)
            y = self.bert.embeddings.word_embeddings.weight @ y.permute(0, 2, 1)
            # y: (batch_size, seq_length, vocab_size)
            y = y.permute(0, 2, 1)
        return y


def tokenize_item(tokenizer: PreTrainedTokenizer, item):
    text = item['text']
    toks = tokenizer(text, add_special_tokens=False).input_ids
    return {
        **item,
        'toks': toks,
        'toks_len': len(toks),
    }


class MaskDataset:
    def __init__(self, pad_token_id: int = 0, max_seq_len: int = 512, mask_token_id: int = 103, min_mask_toks: int = 0, max_mask_toks: int = 10):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.mask_token_id = mask_token_id
        self.min_mask_toks = min_mask_toks
        self.max_mask_toks = max_mask_toks
        self.inds = np.arange(self.max_seq_len)

    def extract_masked_input(self, item: Dict) -> Dict:
        cur_len = item['toks_len']
        max_seq_len = min(cur_len, self.max_seq_len)
        input_ids = item['toks']
        if max_seq_len < cur_len:
            ind_off_max = cur_len - max_seq_len + 1
            ind_off_max = min(ind_off_max, 3)
            ind_off = np.random.randint(0, ind_off_max)
            input_ids = input_ids[ind_off:ind_off + max_seq_len]
        input_ids_masked = input_ids
        mask_toks_num = np.random.randint(self.min_mask_toks, self.max_mask_toks + 1)
        mask_toks_num = min(mask_toks_num, max_seq_len // 2)
        if mask_toks_num > 0:
            mask_inds = np.random.choice(self.inds[:max_seq_len], size=mask_toks_num, replace=False)
            input_ids_masked = np.array(input_ids)
            input_ids_masked[mask_inds] = self.mask_token_id
    
        return {
            **item,
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'input_ids_masked': torch.tensor(input_ids_masked, dtype=torch.long),
        }


class MaskCollateDataset:
    def __init__(self, pad_token_id: int = 0, max_seq_len: int = 512, mask_token_id: int = 103, min_mask_toks: int = 0, max_mask_toks: int = 10):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.mask_token_id = mask_token_id
        self.min_mask_toks = min_mask_toks
        self.max_mask_toks = max_mask_toks
        self.inds = np.arange(self.max_seq_len)

    def extract_masked_input(self, item: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        cur_len = item['toks_len']
        max_seq_len = min(cur_len, self.max_seq_len)
        input_ids = item['toks']
        if max_seq_len < cur_len:
            ind_off_max = cur_len - max_seq_len + 1
            ind_off_max = min(ind_off_max, 3)
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
    
    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids_batch = []
        input_ids_masked_batch = []
        for item in batch:
            input_ids, input_ids_masked = self.extract_masked_input(item)
            input_ids_batch.append(input_ids)
            input_ids_masked_batch.append(input_ids_masked)
        input_ids = nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=self.pad_token_id)
        input_ids_masked = nn.utils.rnn.pad_sequence(input_ids_masked_batch, batch_first=True, padding_value=self.pad_token_id)
        return input_ids, input_ids_masked


def train(dataset: Dataset, model_name: str, share_inout_embeddings: bool, batch_size: int, rank: int = -1, world_size: int = -1):
    '''Training function for each GPU process.

    Args:
        rank (int): The rank of the current process (one per GPU).
        world_size (int): Total number of processes.
    '''
    # Initialize the process group for distributed training
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{rank}')  # Set device to current GPU

    # Create a DistributedSampler to ensure each GPU gets a different mini-batch
    train_sampler = DistributedSampler(dataset)

    # Create a DataLoader with the DistributedSampler
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for distributed training (handled by the sampler)
        sampler=train_sampler,
        num_workers=4, # Sets the number of subprocesses to load data in parallel, avoiding I/O bottlenecks
        pin_memory=True, # Copies data to pinned memory, which is faster to transfer to the GPU.
        prefetch_factor=2 # Sets the number of batches that each worker will prepare in advance.
    )

    # Instantiate the model and move it to the current GPU
    model = MaskedBert(share_inout_embeddings=share_inout_embeddings).to(device)
    # Wrap the model with DDP
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(ddp_model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Example training loop
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Clean up
    dist.destroy_process_group()


def load_masked_wiki_dataset(data_path: Path, model_name: str) -> Tuple[PreTrainedTokenizer, Dataset]:
    tkz = BertModel.from_pretrained(model_name).get_input_embeddings().tokenizer
    wiki_ds_name, wiki_ds_subdir = '20220301.en', 'wikipedia'
    dataset = load_dataset(wiki_ds_subdir, wiki_ds_name, cache_dir=str(data_path))[wiki_ds_subdir]['train']
    dataset = dataset.map(tokenize_item, fn_kwargs={'tokenizer': tkz}, remove_columns=dataset.column_names)
    dataset = dataset.map(MaskDataset(pad_token_id=tkz.pad_token_id, max_seq_len=512, mask_token_id=tkz.mask_token_id, min_mask_toks=0, max_mask_toks=10).extract_masked_input, remove_columns=dataset.column_names)
    return tkz, dataset


def run_training():
    default_data_path = Path(os.path.expandvars('$HOME/data'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-rank', type=int)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--data-path', type=Path, default=default_data_path)
    parser.add_argument('--share-inout-embeddings', action='store_true')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased')
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    local_rank = args.local_rank
    wiki_ds_name, wiki_ds_subdir = '20220301.en', 'wikipedia'

    tokenizer, dataset = load_masked_wiki_dataset(args.data_path, args.model_name)

    # Launch one process per GPU
    mp.spawn(train, args=(dataset, args.model_name, args.share_inout_embeddings, args.batch_size, local_rank, world_size), nprocs=world_size)


if __name__ == '__main__':
    run_training()
