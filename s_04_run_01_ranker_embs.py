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
from mllm.exp.cfg import create_mllm_ranker_cfg
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
    docs_total: int = Field(
        0,
        required=False,
        description='Number of documents to run. If empty or <= 0 then all the documents will be processed.',
        cli=('--docs-total',),
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

    n_docs = len(ds.df_off)
    docs_batch_it = ds.get_docs_batch_iterator(
        batch_size=args.batch_size,
        n_batches=args.docs_total,
        device=device,
    )

    run_info = RunInfo(
        ds_dir_paths=args.ds_dir_paths,
        model_fpath=best_checkpoint_path,
        emb_chunk_size=args.emb_chunk_size,
    )
    run_info_fpath = args.out_ds_path / 'run_info.yaml'
    to_yaml_file(run_info_fpath, run_info)

    model.eval()
    n_it = args.docs_total if args.docs_total is not None and args.docs_total > 0 else n_docs
    pbar = trange(n_it, desc=f'Docs emb inference', unit='doc')
    ind = 0
    docs_embs_fpath = args.out_ds_path / 'docs_embs.npy'
    with open(docs_embs_fpath, 'wb') as f:
        for _ in pbar:
            batch = next(docs_batch_it)
            docs_chunks = batch.gen_tensor()
            docs_embs = model.run_enc_emb(docs_chunks)
            print(docs_chunks.shape, docs_embs.shape)
            df = pd.DataFrame({'ds_ids': batch.ds_ids, 'ds_doc_ids': batch.ds_doc_ids})
            docs_embs = docs_embs.detach().cpu().numpy().astype(np.float32).tobytes('C')
            f.write(docs_embs)
            ids_fpath = args.out_ds_path / f'ids_{ind:04d}.tsv'
            write_tsv(df, ids_fpath)
            ind += 1

    return 0


if __name__ == '__main__':
    def rethrow(e):
        raise e
    run_and_exit(ArgsRunRankerEmbs, main, 'Run Mllm Ranking model inference to form chunks for the next level.', exception_handler=rethrow)


