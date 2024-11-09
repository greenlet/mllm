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

from mllm.data.utils import load_qrels_datasets
from mllm.config.model import create_mllm_ranker_cfg, TokenizerCfg, MllmRankerCfg
from mllm.model.mllm_ranker import MllmRanker, MllmRankerLevel
from mllm.tokenization.chunk_tokenizer import gen_all_tokens, ChunkTokenizer, tokenizer_from_config
from mllm.utils.utils import write_tsv


EMBS_FNAME = 'docs_embs.npy'
IDS_FNAME = 'docs_embs_ids.tsv'
RUN_INFO_FNAME = 'run_info.yaml'


class ArgsGenBertEmbs(BaseModel):
    data_path: Path = Field(
        ...,
        required=True,
        description='Root data path. Must contain subpath `wikipedia/WIKI_DS_NAME` with Wikipedia dataset.',
        cli=('--data-path',),
    )
    wiki_ds_name: str = Field(
        '20200501.en',
        required=False,
        description='Wikipedia dataset name of the format YYYYMMDD.LANG, for example: 20220301.en',
        cli=('--wiki-ds-name',),
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
    max_chunks_per_doc: int = Field(
        10,
        required=True,
        description='Maximum tokens chunks per document.',
        cli=('--max-chunks-per-doc',)
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
    wiki_ds_path: Path
    bert_pretrained_model_name: str
    bert_attn_impl: str
    emb_size: int = 0
    emb_bytes_size: int = 0
    n_docs_written: int = 0
    n_embs_written: int = 0



def tokens_to_chunks(toks: list[int], chunk_size: int, max_chunks: int, pad_tok: int) -> np.ndarray:
    n = len(toks)
    assert n > 0
    nd, nm = divmod(n, chunk_size)
    n_chunks = nd
    if nd == 0  or nm >= 10:
        n_chunks += 1
    n_chunks = min(n_chunks, max_chunks)
    chunks = np.full(n_chunks * chunk_size, pad_tok, dtype=np.int32)
    nc = min(n, len(chunks))
    chunks[:nc] = toks[:nc]
    chunks = chunks.reshape((n_chunks, chunk_size))
    return chunks


def gen_mask(chunks: torch.Tensor, pad_tok: int) -> torch.Tensor:
    mask = chunks == pad_tok
    return mask.to(torch.uint8)


def main(args: ArgsGenBertEmbs) -> int:
    print(args)

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

    print(f'Loading Wikipedia dataset: {args.wiki_ds_name}')
    wiki_ds_subdir = 'wikipedia'
    ds = load_dataset(wiki_ds_subdir, args.wiki_ds_name, beam_runner='DirectRunner', cache_dir=str(args.data_path))
    ds = ds['train']
    n_docs = len(ds)
    print(f'Wikipedia {args.wiki_ds_name} docs: {n_docs}')
    n_docs = min(n_docs, args.max_docs) if args.max_docs > 0 else n_docs
    pbar = trange(n_docs, desc=f'Bert inference', unit='doc')
    run_info = RunInfo(
        wiki_ds_path=args.data_path / wiki_ds_subdir / args.wiki_ds_name,
        bert_pretrained_model_name=bert_pretrained_model_name,
        bert_attn_impl=bert_attn_impl,
    )
    embs_fpath = args.out_ds_path / EMBS_FNAME
    ids_fpath = args.out_ds_path / IDS_FNAME
    run_info_fpath = args.out_ds_path / RUN_INFO_FNAME
    fout = open(embs_fpath, 'wb')
    docs_ids = np.empty(n_docs * args.max_chunks_per_doc, dtype=np.uint32)
    embs_ids = docs_ids.copy()
    docs_off = 0
    chunks, chunks_docs_ids, chunks_embs_ids = [], [], []
    for i in pbar:
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
    fout.close()

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


