import shutil
from datetime import datetime
import itertools as it
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import torch
import torch.utils.tensorboard as tb
from tqdm import trange


from mllm.model.model import CfgMllm, Mllm, create_mllm_cfg
from mllm.tokenization.chunk_tokenizer import gen_ds_fnames, parse_out_subdir, gen_add_doc_tokens, split_doc_embs
from transformers import GPT2Tokenizer


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
    docs_batch_size: Optional[int] = Field(
        3,
        required=False,
        description='Documents batch size. Must be greater or equal than 2.',
        cli=('--docs-batch-size',),
    )
    max_chunks_per_doc: Optional[int] = Field(
        3,
        required=False,
        description='Maximum number of consecutive chunks per document taken in each butch. '
                    'Batch chunk max size will be DOCS_BATCH_SIZE * MAX_CHUNKS_PER_DOC.',
        cli=('--max-chunks-per-doc',),
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


class DocsBatch:
    docs_chunks: dict[int, list[np.ndarray]]
    target_doc_id: int
    target_tokens: list[int]
    pad_tok: int
    emb_chunk_size: int
    docs_chunks_padded: np.ndarray
    target_chunks_padded: np.ndarray
    target_mask: np.ndarray
    _docs_chunks_padded_tf: Optional[torch.Tensor] = None
    _target_chunks_padded_tf: Optional[torch.Tensor] = None
    _target_mask_tf: Optional[torch.Tensor] = None

    def __init__(self, docs_chunks: dict[int, list[np.ndarray]], target_doc_id: int, target_tokens: list[int],
                 pad_tok: int, emb_chunk_size: int):
        self.docs_chunks = docs_chunks
        self.target_doc_id = target_doc_id
        self.target_tokens = target_tokens
        self.pad_tok = pad_tok
        self.emb_chunk_size = emb_chunk_size
        self.calc_np()

    def calc_np(self):
        docs_chunks = []
        target_chunk_off, target_chunk_sz = 0, 0
        for doc_id, chunks in self.docs_chunks.items():
            docs_chunks.extend(chunks)
            if target_chunk_sz == 0:
                target_chunk_off += len(chunks)
                if doc_id == self.target_doc_id:
                    target_chunk_sz = len(chunks)

        target_embs_offsets = split_doc_embs(len(self.target_tokens), self.emb_chunk_size)
        n_target_chunks = len(target_embs_offsets) - 1
        target_chunks = []
        for i in range(n_target_chunks):
            chunk = self.target_tokens[target_embs_offsets[i]:target_embs_offsets[i + 1]]
            target_chunks.append(chunk)

        n_batch_chunks = len(docs_chunks)
        max_chank_sz = max(len(chunk) for chunk in it.chain(docs_chunks, target_chunks))

        docs_chunks_padded = np.full((n_batch_chunks, max_chank_sz), self.pad_tok, dtype=np.int32)
        for i_chunk, chunk in enumerate(docs_chunks):
            docs_chunks_padded[i_chunk, :len(chunk)] = chunk

        target_chunks_padded = np.full((n_target_chunks, max_chank_sz), self.pad_tok, dtype=np.int32)
        for i_chunk, chunk in enumerate(target_chunks):
            target_chunks_padded[i_chunk, :len(chunk)] = chunk

        target_mask = np.full(len(docs_chunks), False, dtype=bool)
        target_mask[target_chunk_off:target_chunk_off + target_chunk_sz] = True

        self.docs_chunks_padded = docs_chunks_padded
        self.target_chunks_padded = target_chunks_padded
        self.target_mask = target_mask

    @property
    def docs_chunks_padded_tf(self) -> torch.Tensor:
        if self._docs_chunks_padded_tf is None:
            self._docs_chunks_padded_tf = torch.from_numpy(self.docs_chunks_padded)
        return self._docs_chunks_padded_tf

    @property
    def target_chunks_padded_tf(self) -> torch.Tensor:
        if self._target_chunks_padded_tf is None:
            self._target_chunks_padded_tf = torch.from_numpy(self.target_chunks_padded)
        return self._target_chunks_padded_tf

    @property
    def target_mask_tf(self) -> torch.Tensor:
        if self._target_mask_tf is None:
            self._target_mask_tf = torch.from_numpy(self.target_mask)
        return self._target_mask_tf


class DsLoader:
    ds_dir_path: Path
    emb_chunk_size: int
    fixed_size: bool
    docs_batch_size: int
    max_chunks_per_doc: int
    pad_tok: int
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
                 pad_tok: int, val_ratio: float = 0.2):
        self.ds_dir_path = ds_dir_path
        self.emb_chunk_size, self.fixed_size = parse_out_subdir(ds_dir_path.name)
        self.docs_batch_size = docs_batch_size
        self.max_chunks_per_doc = max_chunks_per_doc
        self.pad_tok = pad_tok
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
                    offset += chunk_size
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
            # print(i, title_beg_ind, title_end_ind, body_beg_ind, body_end_ind)
            # print(len(ch_tokens), ch_tokens[:20])
            if title_beg_ind >= 0:
                assert 0 < title_beg_ind < title_end_ind
                n = len(res)
                res.extend(ch_tokens[title_beg_ind:title_end_ind])
                # print(f'{n} --> {len(res)}')
            if body_beg_ind >= 0:
                assert 0 < body_beg_ind < body_end_ind
                n = len(res)
                res.extend(ch_tokens[body_beg_ind:body_end_ind])
                # print(f'{n} --> {len(res)}')
        # print(f'res: {len(res)}')
        return res

    def get_batch(self, ind: int, train: bool) -> DocsBatch:
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
        return DocsBatch(
            docs_chunks=docs_chunks, target_doc_id=target_docid, target_tokens=target_tokens,
            pad_tok=self.pad_tok, emb_chunk_size=self.emb_chunk_size
        )


def gen_train_subdir(ds_dir_path: Path) -> str:
    dt_str = datetime.now().strftime('%Y%m%d_%M%H%S')
    subdir = f'{dt_str}-{ds_dir_path.parent.name}-{ds_dir_path.name}'
    return subdir


def main(args: ArgsTrain) -> int:
    print(args)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', model_max_length=100000)
    tok_dict = gen_add_doc_tokens(tokenizer)
    pad_tok = tok_dict['pad'].ind
    ds_loader = DsLoader(args.ds_dir_path, args.docs_batch_size, args.max_chunks_per_doc, pad_tok)
    batch = ds_loader.get_batch(100, train=True)
    docs_chunks, target_chunks, target_mask = batch.docs_chunks_padded_tf, batch.target_chunks_padded_tf, batch.target_mask_tf
    print(f'docs_chunks: {docs_chunks.dtype} {docs_chunks.shape}')
    print(f'target_chunks: {target_chunks.dtype} {target_chunks.shape}')
    print(f'target_mask: {target_mask.dtype} {target_mask.shape}')

    train_subdir = gen_train_subdir(args.ds_dir_path)
    train_path = args.train_root_path / train_subdir
    train_path.mkdir(parents=True, exist_ok=True)
    inp_len = ds_loader.emb_chunk_size + ds_loader.emb_chunk_size // 2 - 1
    print(f'Creating model with vocab size = {len(tokenizer)}')

    model_cfg = create_mllm_cfg(
        n_vocab=len(tokenizer), inp_len=inp_len, d_word_wec=256,
        n_head=8, d_k=32, d_v=32, d_model=256, d_inner=256,
        pad_idx=pad_tok,
    )
    print(model_cfg)
    model = Mllm(model_cfg)

    return 0


if __name__ == '__main__':
    def rethrow(e):
        raise e
    run_and_exit(ArgsTrain, main, 'Train Mllm model.', exception_handler=rethrow)


