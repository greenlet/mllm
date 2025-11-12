import argparse
from functools import partial
import os
from pathlib import Path
import shutil
import traceback
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
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from mllm.model.bert.modeling_bert import BertModel
from mllm.model.losses import EncdecMaskPadItemLoss


class MaskedBert(nn.Module):
    def __init__(self, model_name: str, share_inout_embeddings: bool = True):
        super(MaskedBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
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


class MaskedDataset(Dataset):
    def __init__(self, dataset: Dataset, tkz: PreTrainedTokenizer, max_seq_len: int, min_mask_toks: int, max_mask_toks: int):
        self.dataset = dataset
        # self.dataset = dataset.map(partial(tokenize_item, tkz))
        self.tkz = tkz
        self.len = len(self.dataset)
        self.pad_token_id = tkz.pad_token_id
        self.max_seq_len = max_seq_len
        self.mask_token_id = tkz.mask_token_id
        self.min_mask_toks = min_mask_toks
        self.max_mask_toks = max_mask_toks
        self.inds = np.arange(self.max_seq_len)

    def __len__(self):
        return self.len

    def extract_masked_input(self, item: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        toks = self.tkz(item['text'], add_special_tokens=True).input_ids
        cur_len = len(toks)
        max_seq_len = min(cur_len, self.max_seq_len)
        input_ids = toks
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
    

def collate_masked_batch(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids_batch = []
    input_ids_masked_batch = []
    for item in batch:
        input_ids = item['input_ids']
        input_ids_masked = item['input_ids_masked']
        input_ids_batch.append(input_ids)
        input_ids_masked_batch.append(input_ids_masked)
    input_ids = nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=tkz.pad_token_id)
    input_ids_masked = nn.utils.rnn.pad_sequence(input_ids_masked_batch, batch_first=True, padding_value=tkz.pad_token_id)
    return input_ids, input_ids_masked


def create_dataloader_iter(dataset: Dataset, batch_size: int, num_workers: int, collate_fn: Any):
    while True:
        print(f'Generate Dataloader')
        train_sampler = DistributedSampler(dataset)

        # Create a DataLoader with the DistributedSampler
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle for distributed training (handled by the sampler)
            sampler=train_sampler,
            num_workers=num_workers, # Sets the number of subprocesses to load data in parallel, avoiding I/O bottlenecks
            pin_memory=True, # Copies data to pinned memory, which is faster to transfer to the GPU.
            # prefetch_factor=2, # Sets the number of batches that each worker will prepare in advance.
            collate_fn=collate_fn,
        )
        
        for item in dataloader:
            yield item


def train(rank: int, ds_train: Dataset, ds_val: Dataset, tkz: PreTrainedTokenizer, model_name: str, share_inout_embeddings: bool, batch_size: int, world_size: int = -1):
    '''Training function for each GPU process.

    Args:
        rank (int): The rank of the current process (one per GPU).
        world_size (int): Total number of processes.
    '''
    # Initialize the process group for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # acc = torch.accelerator.current_accelerator()
    # backend = torch.distributed.get_default_backend_for_device(acc)
    backend = 'nccl'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{rank}')  # Set device to current GPU

    train_loader = create_dataloader_iter(
        ds_train,
        batch_size=batch_size,
        num_workers=world_size,
        collate_fn=collate_masked_batch,
    )
    val_loader = create_dataloader_iter(
        ds_val,
        batch_size=batch_size,
        num_workers=world_size,
        collate_fn=collate_masked_batch,
    )

    # Instantiate the model and move it to the current GPU
    model = MaskedBert(model_name=model_name, share_inout_embeddings=share_inout_embeddings).to(device)
    # Wrap the model with DDP
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(ddp_model.parameters())
    criterion = EncdecMaskPadItemLoss(
        msk_tok_id=tkz.mask_token_id,
        spc_tok_ids=[tkz.cls_token_id, tkz.sep_token_id, tkz.pad_token_id],
        reg_weight=0.5,
        msk_weight=1.0,
        spc_weight=0.1,
    )

    if rank == 0:
        print('Starting training...')
        train_path = Path(f'./data/train_distrib_example')
        train_path.mkdir(parents=True, exist_ok=True)
        tbsw = tb.SummaryWriter(log_dir=str(train_path))
    try:
        train_steps, val_steps = 100, 10
        train_it, val_it = iter(train_loader), iter(val_loader)
        val_min_loss = float('inf')
        for epoch in range(10):  # Example: 10 training steps
            ddp_model.train()
            if rank == 0:
                pbar = trange(train_steps, desc=f'Epoch {epoch}', unit='batch')
            else:
                pbar = range(train_steps)
            for _ in pbar:
                input_ids, input_ids_masked = next(train_it)
                input_ids, input_ids_masked = input_ids.to(device), input_ids_masked.to(device)
                optimizer.zero_grad()
                logits = ddp_model(input_ids_masked)
                loss = criterion(logits, input_ids_masked, input_ids)
                loss.backward()
                optimizer.step()

                if rank == 0:
                    pbar.set_postfix({'Train Loss': loss.item()})
            if rank == 0:
                pbar.close()
                tbsw.add_scalar('Train/Loss', loss.item(), epoch)

            # Validation loop
            ddp_model.eval()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            with torch.no_grad():
                val_loss = 0.0
                val_steps = 0
                if rank == 0:
                    pbar = trange(val_steps, desc=f'Epoch {epoch}', unit='batch')
                else:
                    pbar = range(val_steps)
                for _ in pbar:
                    input_ids, input_ids_masked = next(val_it)
                    input_ids, input_ids_masked = input_ids.to(device), input_ids_masked.to(device)
                    logits = ddp_model(input_ids_masked)
                    loss = criterion(logits, input_ids_masked, input_ids)
                    val_loss += loss.item()
                    val_steps += 1
                    if rank == 0:
                        pbar.set_postfix({'Val Loss': loss.item()})
                pbar.close()
                val_avg_loss = val_loss / val_steps
                if rank == 0:
                    print(f'Epoch {epoch}. Validation Loss: {val_avg_loss:.4f}')
                    tbsw.add_scalar('Val/Loss', val_avg_loss, epoch)
            
            if rank == 0:
                tbsw.flush()
                last_fpath = train_path / 'model_last.pt'
                print(f'Save model weights to {last_fpath}')
                torch.save(ddp_model.state_dict(), last_fpath)
                if val_avg_loss < val_min_loss:
                    print(f'New best model found at epoch {epoch}. Val loss: {val_min_loss:.6f}  --> {val_avg_loss:.6f}.')
                    best_fpath = train_path / 'model_best.pt'
                    val_min_loss = val_avg_loss
                    print(f'Save best model weights to {best_fpath}')
                    shutil.copyfile(last_fpath, best_fpath)
    except Exception as e:
        print(e)
        print(traceback.format_stack())

    # Clean up
    dist.destroy_process_group()


def train_v2(rank: int, model_name: str, share_inout_embeddings: bool, batch_size: int, data_path: Path, max_seq_len: int, min_mask_toks: int, max_mask_toks: int, val_split_ratio: float, world_size: int = -1):
    tkz = AutoTokenizer.from_pretrained(model_name)     
    print(tkz)
    ds_train, ds_val = load_masked_wiki_dataset(
        data_path, tkz,
        max_seq_len=max_seq_len,
        min_mask_toks=min_mask_toks,
        max_mask_toks=max_mask_toks,
        val_split_ratio=val_split_ratio,
    )

    '''Training function for each GPU process.

    Args:
        rank (int): The rank of the current process (one per GPU).
        world_size (int): Total number of processes.
    '''
    # Initialize the process group for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{rank}')  # Set device to current GPU

    train_loader = create_dataloader_iter(
        ds_train,
        batch_size=batch_size,
        num_workers=world_size,
        collate_fn=collate_masked_batch,
    )
    val_loader = create_dataloader_iter(
        ds_val,
        batch_size=batch_size,
        num_workers=world_size,
        collate_fn=collate_masked_batch,
    )

    # Instantiate the model and move it to the current GPU
    model = MaskedBert(model_name=model_name, share_inout_embeddings=share_inout_embeddings).to(device)
    # Wrap the model with DDP
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(ddp_model.parameters())
    criterion = EncdecMaskPadItemLoss(
        msk_tok_id=tkz.mask_token_id,
        spc_tok_ids=[tkz.cls_token_id, tkz.sep_token_id, tkz.pad_token_id],
        reg_weight=0.5,
        msk_weight=1.0,
        spc_weight=0.1,
    )

    if rank == 0:
        print('Starting training...')
        train_path = Path(f'./train_distrib_example')
        train_path.mkdir(parents=True, exist_ok=True)
        tbsw = tb.SummaryWriter(log_dir=str(train_path))
    train_steps, val_steps = 100, 10
    train_it, val_it = iter(train_loader), iter(val_loader)
    val_min_loss = float('inf')
    for epoch in range(10):  # Example: 10 training steps
        ddp_model.train()
        if rank == 0:
            pbar = trange(train_steps, desc=f'Epoch {epoch}', unit='batch')
        else:
            pbar = range(train_steps)
        for _ in pbar:
            input_ids, input_ids_masked = next(train_it)
            input_ids, input_ids_masked = input_ids.to(device), input_ids_masked.to(device)
            optimizer.zero_grad()
            logits = ddp_model(input_ids_masked)
            loss = criterion(logits, input_ids_masked, input_ids)
            loss.backward()
            optimizer.step()

            if rank == 0:
                pbar.set_postfix({'Train Loss': loss.item()})
        if rank == 0:
            pbar.close()
            tbsw.add_scalar('Train/Loss', loss.item(), epoch)

        # Validation loop
        ddp_model.eval()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        with torch.no_grad():
            val_loss = 0.0
            val_steps = 0
            if rank == 0:
                pbar = trange(val_steps, desc=f'Epoch {epoch}', unit='batch')
            else:
                pbar = range(val_steps)
            for _ in pbar:
                input_ids, input_ids_masked = next(val_it)
                input_ids, input_ids_masked = input_ids.to(device), input_ids_masked.to(device)
                logits = ddp_model(input_ids_masked)
                loss = criterion(logits, input_ids_masked, input_ids)
                val_loss += loss.item()
                val_steps += 1
                if rank == 0:
                    pbar.set_postfix({'Val Loss': loss.item()})
            pbar.close()
            val_avg_loss = val_loss / val_steps
            if rank == 0:
                print(f'Epoch {epoch}. Validation Loss: {val_avg_loss:.4f}')
                tbsw.add_scalar('Val/Loss', val_avg_loss, epoch)
        
        if rank == 0:
            tbsw.flush()
            last_fpath = train_path / 'model_last.pt'
            print(f'Save model weights to {last_fpath}')
            torch.save(ddp_model.state_dict(), last_fpath)
            if val_avg_loss < val_min_loss:
                print(f'New best model found at epoch {epoch}. Val loss: {val_min_loss:.6f}  --> {val_avg_loss:.6f}.')
                best_fpath = train_path / 'model_best.pt'
                val_min_loss = val_avg_loss
                print(f'Save best model weights to {best_fpath}')
                shutil.copyfile(last_fpath, best_fpath)

    # Clean up
    dist.destroy_process_group()


def load_masked_wiki_dataset(
        data_path: Path, tkz: PreTrainedTokenizer, max_seq_len: int, min_mask_toks: int, max_mask_toks: int, val_split_ratio: float,
        random_seed: Optional[int] = 55,
    ) -> Tuple[Dataset, Dataset]:
    data_path.mkdir(parents=True, exist_ok=True)
    wiki_ds_name, wiki_ds_subdir = '20220301.en', 'wikipedia'
    dataset = load_dataset(wiki_ds_subdir, wiki_ds_name, cache_dir=str(data_path), trust_remote_code=True)['train']

    if random_seed is not None:
        dataset = dataset.shuffle(seed=random_seed)
    n_total = len(dataset)
    n_val = int(n_total * val_split_ratio)
    n_train = n_total - n_val
    ds_train = dataset[:n_train]
    ds_val = dataset[n_train:]

    dataset = MaskedDataset(dataset, tkz, max_seq_len=max_seq_len, min_mask_toks=min_mask_toks, max_mask_toks=max_mask_toks)
    return ds_train, ds_val


def run_training():
    default_data_path = Path(os.path.expandvars('./data'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--data-path', type=Path, default=default_data_path)
    parser.add_argument('--share-inout-embeddings', action='store_true')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased')
    parser.add_argument('--max-seq-len', type=int, default=512)
    parser.add_argument('--min-mask-toks', type=int, default=1)
    parser.add_argument('--max-mask-toks', type=int, default=10)
    parser.add_argument('--val-split-ratio', type=float, default=0.1)

    args = parser.parse_args()
    world_size = torch.cuda.device_count()

    tkz = AutoTokenizer.from_pretrained(args.model_name)
    print(tkz)
    ds_train, ds_val = load_masked_wiki_dataset(
        args.data_path, tkz,
        max_seq_len=args.max_seq_len,
        min_mask_toks=args.min_mask_toks,
        max_mask_toks=args.max_mask_toks,
        val_split_ratio=args.val_split_ratio,
    )
    # Launch one process per GPU
    mp.spawn(train, args=(
        ds_train, ds_val, tkz, args.model_name, args.share_inout_embeddings, args.batch_size, world_size,
    ), nprocs=world_size, join=True)

    # # Launch one process per GPU
    # mp.spawn(train_v2, args=(
    #     args.model_name, args.share_inout_embeddings, args.batch_size, args.data_path, args.max_seq_len, args.min_mask_toks, args.max_mask_toks, args.val_split_ratio, world_size,
    # ), nprocs=world_size, join=True)


if __name__ == '__main__':
    run_training()

