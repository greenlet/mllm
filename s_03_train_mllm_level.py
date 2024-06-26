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
    fpaths = [p for p in ds_dir_path.iterdir() if p.suffix == '.csv']
    n_files = len(fpaths)
    for i in trange(n_files, desc='Processing csv files', unit='file'):
        fpath = fpaths[i]
        df = pd.read_csv(fpath, header=0)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df.sort_values(['docid', 'offset'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.reset_index(drop=False, names='chid', inplace=True)
    return df


class DsLoader:
    ds_dir_path: Path
    docs_batch_size: int
    max_chunks_per_doc: int
    df: pd.DataFrame
    df_doc: pd.DataFrame
    val_ratio: float
    n_docs: int
    n_docs_train: int
    n_docs_val: int
    docids: np.ndarray
    docids_train: np.ndarray
    docids_val: np.ndarray

    def __init__(self, ds_dir_path: Path, docs_batch_size: int, max_chunks_per_doc: int,
                 val_ratio: float = 0.2):
        self.ds_dir_path = ds_dir_path
        self.docs_batch_size = docs_batch_size
        self.max_chunks_per_doc = max_chunks_per_doc
        self.df = read_csv_files(ds_dir_path)
        self.df.set_index(['docid', 'offset'], inplace=True)
        df_doc = self.df.groupby(level=['docid'])
        df_doc = df_doc.agg({'chid': 'count', 'title_tok_num': 'sum', 'body_tok_num': 'sum', 'tok_num': 'sum'})
        df_doc.rename({'chid': 'chunks'}, axis=1, inplace=True)
        self.df_doc = df_doc
        self.val_ratio = val_ratio
        self.docids = self.df_doc.index.to_numpy()
        self.n_docs = len(self.docids)
        self.n_docs_val = int(self.n_docs * self.val_ratio)
        self.n_docs_train = self.n_docs - self.n_docs_val
        self.docids_train = self.docids[:self.n_docs_train]
        self.docids_val = self.docids[self.n_docs_train:]
        # print(self.df)

    def get_batch(self, ind: int, train: bool):
        docids = self.docids_train if train else self.docids_val
        docids = docids[ind:ind + self.docs_batch_size]
        dfd = self.df_doc.loc[docids]
        for docid in docids:
            n_chunks = self.df_doc.loc[docid]['chunks']
            df = self.df.loc[docid]
            print(df)







def main(args: ArgsTrain) -> int:
    print(args)

    docs_batch_size = 3
    max_chunks_per_doc = 3
    ds_loader = DsLoader(args.ds_dir_path, docs_batch_size, max_chunks_per_doc)
    ds_loader.get_batch(1, train=True)

    return 0


if __name__ == '__main__':
    run_and_exit(ArgsTrain, main, 'Train Mllm model.')


