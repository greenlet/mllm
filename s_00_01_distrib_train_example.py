import argparse
import os
from pathlib import Path

from datasets import Dataset, load_dataset
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from mllm.model.bert.modeling_bert import BertModel


class MaskedBert(nn.Module):
    def __init__(self):
        super(MaskedBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(x)
        return out.last_hidden_state


def train(dataset: Dataset, batch_size: int, rank: int = -1, world_size: int = -1):
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
    model = MaskedBert().to(device)
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


def load_masked_wiki_dataset(data_path: Path) -> Dataset:
    wiki_ds_name, wiki_ds_subdir = '20220301.en', 'wikipedia'
    dataset = load_dataset(wiki_ds_subdir, wiki_ds_name, cache_dir=str(data_path))[wiki_ds_subdir]['train']
    return dataset


def run_training():
    default_data_path = Path(os.path.expandvars('$HOME/data'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-rank', type=int)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--data-path', type=Path, default=default_data_path)
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    local_rank = args.local_rank
    wiki_ds_name, wiki_ds_subdir = '20220301.en', 'wikipedia'

    dataset = load_dataset(wiki_ds_subdir, wiki_ds_name, cache_dir=str(args.data_path))[wiki_ds_subdir]['train']

    # Launch one process per GPU
    mp.spawn(train, args=(dataset, args.batch_size, local_rank, world_size), nprocs=world_size)


if __name__ == '__main__':
    run_training()
