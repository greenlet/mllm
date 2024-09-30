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

from mllm.data.dsqrels_embs import DsQrelsEmbs
from mllm.model.mllm_encdec import MllmEncdecLevel
from transformers import GPT2Tokenizer

from mllm.data.utils import load_qrels_datasets
from mllm.config.model import create_mllm_ranker_cfg, TokenizerCfg, MllmRankerCfg, MllmEncdecCfg
from mllm.model.mllm_ranker import MllmRanker
from mllm.tokenization.chunk_tokenizer import gen_all_tokens, ChunkTokenizer, tokenizer_from_config
from mllm.utils.utils import write_tsv


class ArgsRunRankerEmbs(BaseModel):
    ds_dir_path: Path = Field(
        None,
        required=False,
        description='Embeddings dataset path. Must contain docs_embs.npy, docs_ids.tsv, qs_embs.npy, qs_ids.tsv files with'
                    'Embeddings generated from previous step and doc/query ids corresponding to embeddings.',
        cli=('--ds-dir-path',),
    )
    train_dir_path: Path = Field(
        ...,
        required=True,
        description='Path to encdec train directory.',
        cli=('--train-root-path',),
    )
    model_cfg_fpath: Path = Field(
        ...,
        required=True,
        description='Path to encdec model config Yaml file.',
        cli=('--model-cfg-fpath',),
    )
    model_level: int = Field(
        ...,
        required=True,
        description='Model level. 0 - start from tokens and produce embeddins_0. k - start from embeddings from level k - 1 '
                    'and produce embeddings_k.',
        cli=('--model-level',),
    )
    chunk_size: int = Field(
        10,
        required=False,
        description='Number of embedding in a chunk.',
        cli=('--chunk-size',),
    )
    batch_size: int = Field(
        3,
        required=False,
        description='Embeddings chunks batch size for inference.',
        cli=('--batch-size',),
    )
    n_batches: int = Field(
        0,
        required=False,
        description='Number of batches to run. If empty or <= 0 then all the batches will be processed.',
        cli=('--n-batches',),
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
    embs_ds_dir_path: Path
    model_fpath: Path
    embs_chunk_size: int
    n_embs_total: int = 0
    n_embs_chunks_total: int = 0
    n_embs: int = 0
    n_embs_chunks: int = 0


def ids_fname(i: int) -> str:
    return f'ids_{i:04}.tsv'


def main(args: ArgsRunRankerEmbs) -> int:
    print(args)

    device = torch.device(args.device)

    best_checkpoint_path = args.train_dir_path / 'best.pth'
    checkpoint = torch.load(best_checkpoint_path, map_location=device)

    model_cfg = parse_yaml_file_as(MllmEncdecCfg, args.model_cfg_fpath)
    print(model_cfg)
    enc_cfg = model_cfg.encoders[args.model_level]

    ds = DsQrelsEmbs(args.ds_dir_path, args.chunk_size, enc_cfg.d_model, np.float32, device)

    model = MllmEncdecLevel(model_cfg, args.model_level).to(device)
    model.load_state_dict(checkpoint['model'])
    # print(model)

    args.out_ds_path.mkdir(parents=True, exist_ok=True)

    n_embs_total = len(ds.df_docs_ids)
    n_embs_chunks_total = n_embs_total // args.chunk_size + min(n_embs_total % args.chunk_size, 1)
    n_embs_batches_total = n_embs_chunks_total // args.batch_size + min(n_embs_chunks_total % args.batch_size, 1)
    n_embs, n_embs_chunks, n_embs_batches = n_embs_total, n_embs_chunks_total, n_embs_batches_total
    if args.n_batches > 0 and args.n_batches < n_embs_batches_total:
        n_embs_batches = args.n_batches
        n_embs_chunks = n_embs_batches * args.batch_size
        n_embs = n_embs_chunks * args.chunk_size

    run_info = RunInfo(
        embs_ds_dir_path=args.ds_dir_path,
        model_fpath=best_checkpoint_path,
        embs_chunk_size=args.chunk_size,
        n_embs_total=n_embs_total,
        n_embs_chunks_total=n_embs_chunks_total,
        n_embs=n_embs,
        n_embs_chunks=n_embs_chunks,
    )
    run_info_fpath = args.out_ds_path / 'run_info.yaml'
    to_yaml_file(run_info_fpath, run_info)

    model.eval()

    print(f'Processing {n_embs} embeddings, {n_embs_chunks} chunks, {n_embs_batches} batches')
    pbar = trange(n_embs_batches, desc=f'Embs chunks emb inference', unit='doc')
    docs_embs_fpath = args.out_ds_path / 'docs_embs.npy'
    docs_embs_ids, embs_chunks_ids, chunks_off_ids = [], [], []

    view = ds.get_embs_view(args.batch_size)
    batch_it = view.get_batch_iterator(
        n_batches=n_embs_batches, batch_size=args.batch_size, with_queries=False,
    )
    for i in pbar:
        batch = next(batch_it)
        # [n_batch, chunk_size, emb_size]
        docs_embs = batch.get_docs_embs_tensor()
        # [n_batch, emb_size]
        docs_embs = model.run_enc_emb(docs_embs)


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

    df_docs_ids = pd.DataFrame({'doc_emb_id': np.arange(len(ds_ids)), 'ds_id': ds_ids, 'ds_doc_id': ds_doc_ids})
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

    df_qs_ids = pd.DataFrame({'query_emb_id': np.arange(len(ds_ids)), 'ds_id': ds_ids, 'ds_query_id': ds_query_ids})
    qs_ids_fpath = args.out_ds_path / 'qs_ids.tsv'
    print(f'Writing queries ids dataset of size {len(df_qs_ids)} in {qs_ids_fpath}')
    write_tsv(df_qs_ids, qs_ids_fpath)

    qrels_fpath = args.out_ds_path / 'qrels.tsv'
    write_tsv(ds.df_qrels, qrels_fpath)

    return 0


if __name__ == '__main__':
    def rethrow(e):
        raise e
    run_and_exit(ArgsRunRankerEmbs, main, 'Run Mllm Ranking model inference to form chunks for the next level.', exception_handler=rethrow)


