import os
from pathlib import Path
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from pydantic import Field, BaseModel
from pydantic_cli import run_and_exit
from pydantic_yaml import to_yaml_file, parse_yaml_file_as
from torch.optim.optimizer import required
from tqdm import trange
from transformers import BertModel, BertTokenizerFast

from mllm.data.dsqrels import QrelsPlainBatch
from mllm.data.utils import load_qrels_datasets
from mllm.config.model import create_mllm_ranker_cfg, TokenizerCfg, MllmRankerCfg
from mllm.model.mllm_ranker import MllmRanker, MllmRankerLevel
from mllm.tokenization.chunk_tokenizer import gen_all_tokens, ChunkTokenizer, tokenizer_from_config
from mllm.utils.utils import write_tsv


QS_EMBS_FNAME = 'qs_embs.npy'
DOCS_EMBS_FNAME = 'docs_embs.npy'
RUN_INFO_FNAME = 'run_info.yaml'


class ArgsGenBertQrelsEmbs(BaseModel):
    ds_dir_paths: list[Path] = Field(
        [],
        required=True,
        description='Qrels datasets directory paths. Supported datasets: Msmarco, Fever.'
                    'Naming convention: directory name must contain the name of dataset: msmarco, fever. Unknown datasets '
                    'will cause an error and exit.',
        cli=('--ds-dir-paths',),
    )
    out_ds_path: Path = Field(
        ...,
        required=True,
        description='Path to a directory where embeddings generated will be stored',
        cli=('--out-ds-path',),
    )
    tokens_chunk_size: int = Field(
        512,
        required=True,
        description='Number of tokens in input chunk.',
        cli=('--tokens-chunk-size',)
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
    batch_size: int = Field(
        3,
        required=False,
        description='Tokens chunks batch size for inference.',
        cli=('--batch-size',),
    )
    device: str = Field(
        'cpu',
        required=False,
        description='Device to run inference on. Can have values: "cpu", "cuda"',
        cli=('--device',)
    )


class RunInfo(BaseModel):
    ds_dir_paths: list[Path]
    bert_pretrained_model_name: str
    bert_attn_impl: str
    emb_size: int = 0
    emb_bytes_size: int = 0
    n_docs_written: int = 0
    n_qs_written: int = 0


def main(args: ArgsGenBertQrelsEmbs) -> int:
    print(args)

    assert args.ds_dir_paths, '--ds-dir-paths is expected to list at least one Qrels datsaset'
    device = torch.device(args.device)
    args.out_ds_path.mkdir(parents=True, exist_ok=True)

    bert_pretrained_model_name = 'bert-base-uncased'
    bert_attn_impl = 'sdpa'
    model = BertModel.from_pretrained(bert_pretrained_model_name, torch_dtype=torch.float32, attn_implementation=bert_attn_impl)
    model.to(device)
    model.eval()
    model.config.max_position_embeddings = args.tokens_chunk_size
    tokenizer = BertTokenizerFast.from_pretrained(bert_pretrained_model_name)
    pad_tok = tokenizer.pad_token_id

    print(f'Loading Qrels datasets: {args.ds_dir_paths}')
    ch_tkz = ChunkTokenizer({}, tokenizer, n_emb_tokens=args.tokens_chunk_size, fixed_size=True)
    ds = load_qrels_datasets(args.ds_dir_paths, ch_tkz, args.tokens_chunk_size, device)
    print(ds)

    n_docs_total = len(ds.df_off)
    n_docs = args.n_docs if args.n_docs > 0 else n_docs_total
    n_docs = min(n_docs, n_docs_total)
    docs_it = ds.get_docs_iterator(n_docs=n_docs)

    n_qs_total = len(ds.df_qs)
    n_qs = args.n_qs if args.n_qs > 0 else n_qs_total
    n_qs = min(n_qs, n_qs_total)
    qs_it = ds.get_qs_iterator(n_qs=n_qs)

    run_info = RunInfo(
        ds_dir_paths=args.ds_dir_paths,
        bert_pretrained_model_name=bert_pretrained_model_name,
        bert_attn_impl=bert_attn_impl,
        emb_size=args.tokens_chunk_size,
        emb_bytes_size=args.tokens_chunk_size * 4,
    )

    def to_tensor(chunks: list[np.ndarray]) -> torch.Tensor:
        chunks = np.stack(chunks, axis=0)
        t = torch.from_numpy(chunks)
        return t.to(device)

    docs_embs_fpath = args.out_ds_path / DOCS_EMBS_FNAME
    qs_embs_fpath = args.out_ds_path / QS_EMBS_FNAME
    run_info_fpath = args.out_ds_path / RUN_INFO_FNAME
    to_yaml_file(run_info_fpath, run_info)

    print(f'Processing {n_docs} documents')
    pbar = trange(n_docs, desc=f'Docs emb inference', unit='doc')
    batch_toks_np = np.full((args.batch_size, args.tokens_chunk_size), pad_tok, dtype=np.int32)
    batch_masks_np = np.full((args.batch_size, args.tokens_chunk_size), 0, dtype=np.int32)
    batch_ind = 0
    with open(docs_embs_fpath, 'wb') as f:
        for i, _ in enumerate(pbar):
            dc = next(docs_it)
            batch_toks_np[batch_ind, :len(dc.doc_tokens)] = dc.doc_tokens
            batch_masks_np[batch_ind, :len(dc.doc_tokens)] = 1
            batch_ind += 1
            if batch_ind == args.batch_size or i == n_docs - 1:
                batch_toks_t = torch.from_numpy(batch_toks_np[:batch_ind]).to(device)
                batch_masks_t = torch.from_numpy(batch_masks_np[:batch_ind]).to(device)
                out = model(batch_toks_t, batch_masks_t)
                last_hidden_state, pooler_output = out['last_hidden_state'], out['pooler_output']
                embs = pooler_output
                embs = embs.detach().cpu().numpy().astype(np.float32).tobytes('C')
                f.write(embs)
                run_info.n_docs_written += batch_ind
                batch_ind = 0
                batch_toks_np.fill(pad_tok)
                batch_masks_np.fill(0)
    to_yaml_file(run_info_fpath, run_info)

    print(f'Processing {n_qs} queries')
    pbar = trange(n_qs, desc=f'Queries emb inference', unit='query')
    with open(qs_embs_fpath, 'wb') as f:
        for i, _ in enumerate(pbar):
            qc = next(qs_it)
            batch_toks_np[batch_ind, :len(qc.query_tokens)] = qc.query_tokens
            batch_masks_np[batch_ind, :len(qc.query_tokens)] = 1
            batch_ind += 1
            if batch_ind == args.batch_size or i == n_qs - 1:
                batch_toks_t = torch.from_numpy(batch_toks_np[:batch_ind]).to(device)
                batch_masks_t = torch.from_numpy(batch_masks_np[:batch_ind]).to(device)
                out = model(batch_toks_t, batch_masks_t)
                last_hidden_state, pooler_output = out['last_hidden_state'], out['pooler_output']
                embs = pooler_output
                embs = embs.detach().cpu().numpy().astype(np.float32).tobytes('C')
                f.write(embs)
                run_info.n_qs_written += batch_ind
                batch_ind = 0
                batch_toks_np.fill(pad_tok)
                batch_masks_np.fill(0)
    to_yaml_file(run_info_fpath, run_info)

    return 0


if __name__ == '__main__':
    def rethrow(e):
        raise e
    run_and_exit(ArgsGenBertQrelsEmbs, main, 'Run Bert model embeddings inference for Wikipedia dataset.', exception_handler=rethrow)


