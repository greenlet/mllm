import os
from pathlib import Path
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pydantic import Field, BaseModel
from pydantic_cli import run_and_exit
from pydantic_yaml import to_yaml_file
from tqdm import trange
from transformers import GPT2Tokenizer

from mllm.data.utils import load_qrels_datasets
from mllm.config.model import create_mllm_ranker_cfg
from mllm.model.mllm_ranker import MllmRanker
from mllm.tokenization.chunk_tokenizer import gen_all_tokens, ChunkTokenizer
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
    emb_chunk_size: Optional[int] = Field(
        100,
        required=False,
        description='Number of tokens in chunk converted to a single embedding vector.',
        cli=('--emb-chunk-size',),
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

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', model_max_length=10000)
    tok_dict = gen_all_tokens(tokenizer)
    ch_tkz = ChunkTokenizer(tok_dict, tokenizer, n_emb_tokens=args.emb_chunk_size, fixed_size=True)
    pad_tok = tok_dict['pad'].ind

    ds = load_qrels_datasets(args.ds_dir_paths, ch_tkz, args.emb_chunk_size, device)
    print(ds)

    print(f'Creating model with vocab size = {len(tokenizer)}')
    model_cfg = create_mllm_ranker_cfg(
        n_vocab=len(tokenizer), inp_len=args.emb_chunk_size, d_word_wec=256,
        n_levels=1, enc_n_layers=1, dec_n_layers=1,
        n_heads=8, d_k=32, d_v=32, d_model=256, d_inner=1024,
        pad_idx=pad_tok, dropout_rate=0.0, enc_with_emb_mat=True,
    )
    print(model_cfg)
    model = MllmRanker(model_cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    # print(model)

    args.out_ds_path.mkdir(parents=True, exist_ok=True)

    n_docs_total = len(ds.df_off)
    n_docs = args.n_docs if args.n_docs is not None and args.n_docs > 0 else n_docs_total
    n_docs = min(n_docs, n_docs_total)
    docs_chunks_it = ds.get_docs_chunks_iterator(
        n_docs=n_docs,
    )

    run_info = RunInfo(
        ds_dir_paths=args.ds_dir_paths,
        model_fpath=best_checkpoint_path,
        emb_chunk_size=args.emb_chunk_size,
    )
    run_info_fpath = args.out_ds_path / 'run_info.yaml'
    to_yaml_file(run_info_fpath, run_info)

    def to_tensor(chunks: list[np.ndarray]) -> torch.Tensor:
        chunks = np.stack(chunks, axis=0)
        t = torch.from_numpy(chunks)
        return t.to(device)

    model.eval()
    pbar = trange(n_docs, desc=f'Docs emb inference', unit='doc')
    docs_embs_fpath = args.out_ds_path / 'docs_embs.npy'
    ds_ids, ds_doc_ids, docs_chunks = [], [], []
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

    df_ids = pd.DataFrame({'ds_ids': ds_ids, 'ds_doc_ids': ds_doc_ids})
    ids_fpath = args.out_ds_path / 'ids.tsv'
    print(f'Writing ids ds of size {len(df_ids)} in {ids_fpath}')
    write_tsv(df_ids, ids_fpath)

    return 0


if __name__ == '__main__':
    def rethrow(e):
        raise e
    run_and_exit(ArgsRunRankerEmbs, main, 'Run Mllm Ranking model inference to form chunks for the next level.', exception_handler=rethrow)


