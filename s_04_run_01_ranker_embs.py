import os
from pathlib import Path
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pydantic import Field, BaseModel
from pydantic_cli import run_and_exit
from pydantic_yaml import to_yaml_file, parse_yaml_file_as
from tqdm import trange
from transformers import GPT2Tokenizer

from mllm.data.utils import load_qrels_datasets
from mllm.config.model import create_mllm_ranker_cfg, TokenizerCfg, MllmRankerCfg
from mllm.model.mllm_ranker import MllmRanker
from mllm.tokenization.chunk_tokenizer import gen_all_tokens, ChunkTokenizer, tokenizer_from_config
from mllm.utils.utils import write_tsv


class ArgsRunRankerEmbs(BaseModel):
    ds_dir_paths: list[Path] = Field(
        [],
        required=True,
        description='Qrels datasets directory paths. Supported datasets: Msmarco, Fever.'
                    'Naming convention: directory name must contain the name of dataset: msmarco, fever. Unknown datasets '
                    'will cause an error and exit.',
        cli=('--ds-dir-paths',),
    )
    train_root_path: Path = Field(
        ...,
        required=True,
        description='Path to train root directory. Used for loading model weights from subdirectory of interest.',
        cli=('--train-root-path',),
    )
    train_subdir: str = Field(
        '',
        required=True,
        description='Train subdirectory. Must be name of TRAIN_ROOT_PATH subdirectory where model weights are stored (in a file "best.pth").',
        cli=('--train-subdir',)
    )
    tokenizer_cfg_fpath: Path = Field(
        ...,
        required=True,
        description='Path to tokenizer config Yaml file.',
        cli=('--tokenizer-cfg-fpath',),
    )
    model_cfg_fpath: Path = Field(
        ...,
        required=True,
        description='Path to ranker model config Yaml file.',
        cli=('--model-cfg-fpath',),
    )
    batch_size: int = Field(
        3,
        required=False,
        description='Documents batch size for inference.',
        cli=('--batch-size',),
    )
    n_docs: int = Field(
        0,
        required=False,
        description='Number of docs to run. If empty or <= 0 then all the docs will be processed.',
        cli=('--n-docs',),
    )
    n_qs: int = Field(
        0,
        required=False,
        description='Number of queries to run. If empty or <= 0 then all the queries will be processed.',
        cli=('--n-qs',),
    )
    device: str = Field(
        'cpu',
        required=False,
        description='Device to run inference on. Can have values: "cpu", "cuda"',
        cli=('--device',)
    )
    out_ds_path: Path = Field(
        ...,
        required=True,
        description='Path to a directory where embeddings generated will be stored',
        cli=('--out-ds-path',),
    )


class RunInfo(BaseModel):
    ds_dir_paths: list[Path]
    model_fpath: Path
    emb_chunk_size: int
    n_docs: int = 0
    n_docs_chunks: int = 0
    n_qs: int = 0
    n_qs_chunks: int = 0



def ids_fname(i: int) -> str:
    return f'ids_{i:04}.tsv'


def main(args: ArgsRunRankerEmbs) -> int:
    print(args)

    assert args.ds_dir_paths, '--ds-dir-paths is expected to list at least one Qrels datsaset'

    device = torch.device(args.device)

    train_path = args.train_root_path / args.train_subdir
    print(f'train_path: {train_path}')

    best_checkpoint_path = train_path / 'best.pth'
    checkpoint = torch.load(best_checkpoint_path, map_location=device)

    tkz_cfg = parse_yaml_file_as(TokenizerCfg, args.tokenizer_cfg_fpath)
    tokenizer = tokenizer_from_config(tkz_cfg)

    model_cfg = parse_yaml_file_as(MllmRankerCfg, args.model_cfg_fpath)
    n_emb_tokens = model_cfg.encoders[0].inp_len
    print(model_cfg)

    ch_tkz = ChunkTokenizer(tkz_cfg.custom_tokens, tokenizer, n_emb_tokens=n_emb_tokens, fixed_size=True)
    pad_tok = tkz_cfg.custom_tokens['pad'].ind

    ds = load_qrels_datasets(args.ds_dir_paths, ch_tkz, n_emb_tokens, device)
    print(ds)

    print(f'Creating model with vocab size = {len(tokenizer)}')
    model = MllmRanker(model_cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    # print(model)

    args.out_ds_path.mkdir(parents=True, exist_ok=True)

    n_docs_total = len(ds.df_off)
    n_docs = args.n_docs if args.n_docs > 0 else n_docs_total
    n_docs = min(n_docs, n_docs_total)
    docs_chunks_it = ds.get_docs_chunks_iterator(n_docs=n_docs)

    n_qs_total = len(ds.df_qs)
    n_qs = args.n_qs if args.n_qs > 0 else n_qs_total
    n_qs = min(n_qs, n_qs_total)
    qs_chunks_it = ds.get_qs_chunks_iterator(n_qs=n_qs)

    run_info = RunInfo(
        ds_dir_paths=args.ds_dir_paths,
        model_fpath=best_checkpoint_path,
        emb_chunk_size=n_emb_tokens,
        n_docs=n_docs,
        n_qs=n_qs,
    )
    run_info_fpath = args.out_ds_path / 'run_info.yaml'
    to_yaml_file(run_info_fpath, run_info)

    def to_tensor(chunks: list[np.ndarray]) -> torch.Tensor:
        chunks = np.stack(chunks, axis=0)
        t = torch.from_numpy(chunks)
        return t.to(device)

    model.eval()

    print(f'Processing {n_docs} documents')
    pbar = trange(n_docs, desc=f'Docs emb inference', unit='doc')
    docs_embs_fpath = args.out_ds_path / 'docs_embs.npy'
    ds_ids, ds_doc_ids, docs_chunks = [], [], []
    n_docs_chunks = 0
    with open(docs_embs_fpath, 'wb') as f:
        for i, _ in enumerate(pbar):
            dc = next(docs_chunks_it)
            n_chunks = len(dc.doc_chunks)
            ds_ids.extend([dc.ds_id] * n_chunks)
            ds_doc_ids.extend([dc.ds_doc_id] * n_chunks)
            docs_chunks.extend(dc.doc_chunks)

            while len(docs_chunks) >= args.batch_size or len(docs_chunks) > 0 and i == n_docs - 1:
                batch, docs_chunks = docs_chunks[:args.batch_size], docs_chunks[args.batch_size:]
                batch = to_tensor(batch)
                docs_embs = model.run_enc_emb(batch)
                docs_embs = docs_embs.detach().cpu().numpy().astype(np.float32).tobytes('C')
                f.write(docs_embs)
                n_docs_chunks += len(batch)

    run_info.n_docs_chunks = n_docs_chunks
    to_yaml_file(run_info_fpath, run_info)

    df_docs_ids = pd.DataFrame({'ds_ids': ds_ids, 'ds_doc_ids': ds_doc_ids})
    docs_ids_fpath = args.out_ds_path / 'docs_ids.tsv'
    print(f'Writing docs ids dataset of size {len(df_docs_ids)} in {docs_ids_fpath}')
    write_tsv(df_docs_ids, docs_ids_fpath)
    del ds_ids
    del ds_doc_ids
    del df_docs_ids

    print(f'Processing {n_qs} queries')
    pbar = trange(n_qs, desc=f'Queries emb inference', unit='query')
    qs_embs_fpath = args.out_ds_path / 'qs_embs.npy'
    ds_ids, ds_query_ids, qs_chunks = [], [], []
    n_qs_chunks = 0
    with open(qs_embs_fpath, 'wb') as f:
        for i, _ in enumerate(pbar):
            qc = next(qs_chunks_it)
            n_chunks = len(qc.query_chunks)
            ds_ids.extend([qc.ds_id] * n_chunks)
            ds_query_ids.extend([qc.ds_query_id] * n_chunks)
            qs_chunks.extend(qc.query_chunks)

            while len(qs_chunks) >= args.batch_size or len(qs_chunks) > 0 and i == n_qs - 1:
                batch, qs_chunks = qs_chunks[:args.batch_size], qs_chunks[args.batch_size:]
                batch = to_tensor(batch)
                docs_embs = model.run_enc_emb(batch)
                docs_embs = docs_embs.detach().cpu().numpy().astype(np.float32).tobytes('C')
                f.write(docs_embs)
                n_qs_chunks += len(batch)

    run_info.n_qs_chunks = n_qs_chunks
    to_yaml_file(run_info_fpath, run_info)

    df_qs_ids = pd.DataFrame({'ds_ids': ds_ids, 'ds_query_ids': ds_query_ids})
    qs_ids_fpath = args.out_ds_path / 'qs_ids.tsv'
    print(f'Writing queries ids dataset of size {len(df_qs_ids)} in {qs_ids_fpath}')
    write_tsv(df_qs_ids, qs_ids_fpath)

    return 0


if __name__ == '__main__':
    def rethrow(e):
        raise e
    run_and_exit(ArgsRunRankerEmbs, main, 'Run Mllm Ranking model inference to form chunks for the next level.', exception_handler=rethrow)


