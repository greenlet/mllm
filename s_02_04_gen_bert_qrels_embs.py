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
    max_docs: int = Field(
        0,
        required=False,
        description='Maximum number of Wikipedia documents to process. If MAX_DOCS <= 0, all documents will be processed.',
        cli=('--max-docs',),
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

    ds_view = ds.get_view_plain_dids(args.batch_size)

    n_total = len(ds_view)
    if args.max_docs > 0:
        n_total = args.max_docs
    n_batches = n_total // args.batch_size + min(n_total % args.batch_size, 1)
    batch_it = ds_view.get_batch_iterator(n_batches=n_batches)

    run_info = RunInfo(
        ds_dir_paths=args.ds_dir_paths,
        bert_pretrained_model_name=bert_pretrained_model_name,
        bert_attn_impl=bert_attn_impl,
        emb_size=args.tokens_chunk_size,
        emb_bytes_size=args.tokens_chunk_size * 4,
    )

    docs_embs_fpath = args.out_ds_path / DOCS_EMBS_FNAME
    qs_embs_fpath = args.out_ds_path / QS_EMBS_FNAME
    run_info_fpath = args.out_ds_path / RUN_INFO_FNAME
    docs_fout = open(docs_embs_fpath, 'wb')
    qs_fout = open(qs_embs_fpath, 'wb')
    pbar = trange(n_batches, desc=f'Bert inference', unit='batch')
    for ib in pbar:
        batch: QrelsPlainBatch = next(batch_it)
        doc = ds[i]
        title, text = doc['title'], doc['text']
        doc_txt = f'{title} {text}'
        doc_toks = tokenizer(doc_txt)['input_ids']
        doc_chunks = tokens_to_chunks(doc_toks, args.tokens_chunk_size, args.max_chunks_per_doc, pad_tok)
        chunks.extend(list(doc_chunks))
        chunks_docs_ids.extend([i] * len(doc_chunks))
        chunks_embs_ids.extend(list(range(len(doc_chunks))))
        n_chunks = len(chunks)
        if n_chunks >= args.batch_size or n_chunks > 0 and i == n_docs - 1:
            chunks_batch, chunks = chunks[:n_chunks], chunks[n_chunks:]
            docs_ids_batch, chunks_docs_ids = chunks_docs_ids[:n_chunks], chunks_docs_ids[n_chunks:]
            embs_ids_batch, chunks_embs_ids = chunks_embs_ids[:n_chunks], chunks_embs_ids[n_chunks:]
            chunks_batch = np.stack(chunks_batch)
            chunks_batch = torch.tensor(chunks_batch, dtype=torch.int32, device=device)
            masks_batch = gen_mask(chunks_batch, pad_tok)
            out = model(chunks_batch, masks_batch)
            last_hidden_state, pooler_output = out['last_hidden_state'], out['pooler_output']
            chunks_out = pooler_output
            # print(chunks_batch.shape, last_hidden_state.shape, chunks_out.shape)
            embs = chunks_out
            emb_size = embs.shape[-1]
            embs = chunks_out.detach().cpu().numpy().astype(np.float32).tobytes('C')
            fout.write(embs)
            run_info.emb_size = emb_size
            run_info.emb_bytes_size = run_info.emb_size * 4
            run_info.n_docs_written = docs_ids_batch[-1] + 1
            run_info.n_embs_written += n_chunks
            docs_off_next = docs_off + n_chunks
            docs_ids[docs_off:docs_off_next] = docs_ids_batch
            embs_ids[docs_off:docs_off_next] = embs_ids_batch
            docs_off = docs_off_next

    pbar.close()
    docs_fout.close()
    qs_fout.close()

    docs_ids, embs_ids = docs_ids[:docs_off], embs_ids[:docs_off]
    df_ids = pd.DataFrame({'doc_id': docs_ids, 'emb_id': embs_ids})
    print(f'Write {len(df_ids)} rows to {ids_fpath}')
    write_tsv(df_ids, ids_fpath)
    print(f'Run info: {run_info}')
    to_yaml_file(run_info_fpath, run_info)

    return 0


if __name__ == '__main__':
    def rethrow(e):
        raise e
    run_and_exit(ArgsGenBertEmbs, main, 'Run Bert model embeddings inference for Wikipedia dataset.', exception_handler=rethrow)


