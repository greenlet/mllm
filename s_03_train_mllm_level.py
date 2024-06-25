import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import torch
import torch.utils.tensorboard as tb
from tqdm import trange


from mllm.model.model import CfgMllm, Mllm


class ArgsTrain(BaseModel):
    ds_dir_path: Path = Field(
        None,
        required=False,
        description='Dataset directory path. Must contain .csv and .np files with tokenized text.',
        cli=('--ds-dir-path',),
    )
    train_root_path: Path = Field(
        ...,
        required=True,
        description='Path to train root directory. New train subdirectory will be created within each new run.',
        cli=('--train-root-path',),
    )
    batch_size: Optional[int] = Field(
        None,
        required=False,
        description='Batch size. Must be greater or equal than 2.',
        cli=('--batch-size',),
    )
    device: str = Field(
        'cpu',
        required=False,
        description='Device to run training on. Can have values: "cpu", "cuda"',
        cli=('--device',)
    )
    epochs: Optional[int] = Field(
        None,
        required=False,
        description='Number of training epochs.',
        cli=('--epochs',),
    )
    learning_rate: float = Field(
        0.001,
        required=False,
        description='Initial learning rate of the training process.',
        cli=('--learning-rate',)
    )
    train_epoch_steps: Optional[int] = Field(
        None,
        required=False,
        description='Number of training steps per epoch.',
        cli=('--train-epoch-steps',),
    )
    val_epoch_steps: Optional[int] = Field(
        None,
        required=False,
        description='Number of validation steps per epoch.',
        cli=('--val-epoch-steps',),
    )


def read_csv_files(ds_dir_path: Path) -> pd.DataFrame:
    dfs = []
    for fpath in ds_dir_path.iterdir():
        if fpath.suffix == '.csv':
            df = pd.read_csv(fpath, header=0)
            dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df.sort_values(['docid', 'offset'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.reset_index(drop=False, names='chid', inplace=True)
    return df


class DsLoader:
    ds_dir_path: Path
    df: pd.DataFrame
    batch_size: int
    val_ratio: float
    n_total: int
    n_train: int
    n_val: int
    chid_train: np.ndarray
    chid_val: np.ndarray

    def __init__(self, ds_dir_path: Path, batch_size: int, val_ratio: float = 0.2):
        self.ds_dir_path = ds_dir_path
        self.df = read_csv_files(ds_dir_path)
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.n_total = len(self.df)
        self.n_val = int(self.n_total * self.val_ratio)
        self.n_train = self.n_total - self.n_val
        chid = self.df.chid.to_numpy()
        np.random.shuffle(chid)
        self.chid_train = chid[:self.n_train]
        self.chid_val = chid[self.n_train:]


def main(args: ArgsTrain) -> int:
    print(args)

    ds_loader = DsLoader(args.ds_dir_path, args.batch_size)

    return 0


if __name__ == '__main__':
    run_and_exit(ArgsTrain, main, 'Train Mllm model.')


