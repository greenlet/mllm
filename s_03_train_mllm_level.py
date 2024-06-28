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
from mllm.tokenization.chunk_tokenizer import gen_ds_fnames, parse_out_subdir


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


def read_ds_files(ds_dir_path: Path) -> pd.DataFrame:
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
    emb_chunk_size: int
    fixed_size: bool
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
    _tokens_cache: dict[tuple[int, int], np.ndarray]
    _max_cache_size: int = 3

    def __init__(self, ds_dir_path: Path, docs_batch_size: int, max_chunks_per_doc: int,
                 val_ratio: float = 0.2):
        self.ds_dir_path = ds_dir_path
        self.emb_chunk_size, self.fixed_size = parse_out_subdir(ds_dir_path.name)
        self.docs_batch_size = docs_batch_size
        self.max_chunks_per_doc = max_chunks_per_doc
        self.df = read_ds_files(ds_dir_path)
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
        self._tokens_cache = {}
        # print(self.df)

    def _prune_cache(self):
        # Relying on dict's property keep keys/values sorted in order of addition
        if len(self._tokens_cache) > self._max_cache_size:
            keys = list(self._tokens_cache.keys())
            cache = self._tokens_cache
            self._tokens_cache = {k:cache[k] for k in keys[-self._max_cache_size:]}

    def _load_tokens(self, doc_id_min: int, doc_id_max: int) -> np.ndarray:
        doc_ids = doc_id_min, doc_id_max
        tokens = self._tokens_cache.get(doc_ids)
        if tokens is None:
            _, tokens_fname, chunk_sizes_fname = gen_ds_fnames(doc_id_min, doc_id_max)
            tokens_fpath, chunk_sizes_fpath = self.ds_dir_path / tokens_fname, self.ds_dir_path / chunk_sizes_fname
            tokens = np.fromfile(tokens_fpath, dtype=np.int32)
            if self.fixed_size:
                tokens = tokens.reshape((-1, self.emb_chunk_size))
            else:
                assert chunk_sizes_fpath.exists(), f'Chunk size is not fixed. File {chunk_sizes_fpath} is not found.'
                chunk_sizes = np.fromfile(chunk_sizes_fpath, dtype=np.int32)
                n_chunks = len(chunk_sizes)
                tokens_list = [None] * n_chunks
                offset = 0
                for i_chunk in range(n_chunks):
                    chunk_size = chunk_sizes[i_chunk]
                    tokens_list[i_chunk] = tokens[offset:offset + chunk_size]
                tokens = tokens_list
            self._tokens_cache[doc_ids] = tokens
            self._prune_cache()
            assert doc_ids in self._tokens_cache and len(self._tokens_cache) <= self._max_cache_size
        return tokens

    def _extract_content_tokens(self, df_ch: pd.DataFrame, chunks: list[np.ndarray]) -> list[int]:
        res = []
        for i in range(len(df_ch)):
            ch_row = df_ch.iloc[i]
            ch_tokens = chunks[i]
            title_beg_ind, title_end_ind = ch_row['title_beg_ind'], ch_row['title_end_ind']
            body_beg_ind, body_end_ind = ch_row['body_beg_ind'], ch_row['body_end_ind']
            print(i, title_beg_ind, title_end_ind, body_beg_ind, body_end_ind)
            print(len(ch_tokens))
            if title_beg_ind >= 0:
                assert title_end_ind > title_beg_ind
                res.extend(ch_tokens[title_beg_ind:title_end_ind])
            if body_beg_ind >= 0:
                assert body_end_ind > body_beg_ind
                res.extend(ch_tokens[body_beg_ind:body_end_ind])
        return res

    def get_batch(self, ind: int, train: bool) -> tuple[dict[int, list[np.ndarray]], int, list[int]]:
        docids = self.docids_train if train else self.docids_val
        docids = docids[ind:ind + self.docs_batch_size]
        df_doc = self.df_doc.loc[docids]
        docs_chunks = {}
        target_tokens = []
        target_docid = np.random.choice(docids)
        for docid in docids:
            n_chunks = df_doc.loc[docid]['chunks']
            df = self.df.loc[docid]
            # print(df)
            i_chunk = 0
            if n_chunks > self.max_chunks_per_doc:
                i_chunk = np.random.randint(n_chunks - self.max_chunks_per_doc)
            df = df.iloc[i_chunk:i_chunk + self.max_chunks_per_doc]
            doc_id_min, doc_id_max = df['doc_id_min'].iloc[0], df['doc_id_max'].iloc[0]

            tokens = self._load_tokens(doc_id_min, doc_id_max)
            chunks = []
            for _, row in df.iterrows():
                chunk_tokens = tokens[row['doc_id_off']]
                chunks.append(chunk_tokens)

            docs_chunks[docid] = chunks

            if docid == target_docid:
                target_tokens = self._extract_content_tokens(df, chunks)
        return docs_chunks, target_docid, target_tokens


def main(args: ArgsTrain) -> int:
    print(args)

    docs_batch_size = 3
    max_chunks_per_doc = 3
    ds_loader = DsLoader(args.ds_dir_path, docs_batch_size, max_chunks_per_doc)
    docs_chunks, target_doc_id, target_chunks = ds_loader.get_batch(100, train=True)
    print(target_chunks)

    return 0


if __name__ == '__main__':
    def rethrow(e):
        raise e
    run_and_exit(ArgsTrain, main, 'Train Mllm model.', exception_handler=rethrow)


