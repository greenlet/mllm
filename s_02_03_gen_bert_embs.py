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
    tokens_chunk_size: str = Field(
        512,
        required=True,
        description='Number of tokens in input chunk.',
        cli=('--tokens-chunk-size',)
    )
    max_chunks_per_doc: str = Field(
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


def split_toks(toks: list[int], chunk_sz: int, max_chunks: int, pad_tok: int) -> np.ndarray:
    n = len(toks)
    nd, nm = divmod(n, chunk_sz)
    if nd == 0:
        res = np.zeros(chunk_sz, dtype=np.int32)
        res[:nm] = toks
        return res
    chunks = []
    n_chunks = nd + min(nm, 1)
    for i in range(n_chunks):
        chunk = toks[i * chunk_sz:(i + 1) * chunk_sz]
        if len(chunk) < chunk_sz:
            chunk += [pad_tok] * (chunk_sz - len(chunk))
        chunks.append(chunk)
        if len(chunks) == max_chunks:
            break

    return np.array(chunks, dtype=np.int32)


def main(args: ArgsGenBertEmbs) -> int:
    print(args)

    device = torch.device(args.device)
    args.out_ds_path.mkdir(parents=True, exist_ok=True)

    model = BertModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float32, attn_implementation="sdpa")
    model.to(device)
    model.eval()
    model.config.max_position_embeddings = args.tokens_chunk_size
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    print(f'Loading Wikipedia dataset: {args.wiki_ds_name}')
    ds = load_dataset('wikipedia', args.wiki_ds_name, beam_runner='DirectRunner', cache_dir=str(args.data_path))
    ds = ds['train']
    n_docs = len(ds)
    print(f'Wikipedia {args.wiki_ds_name} docs: {n_docs}')
    n_docs = min(n_docs, args.max_docs) if args.max_docs > 0 else n_docs
    pbar = trange(n_docs, desc=f'Bert inference', unit='doc')
    chunks = []
    for i in pbar:
        doc = ds[i]
        title, text = doc['title'], doc['test']
        doc_txt = f'{title} {text}'
        doc_toks = tokenizer(doc_txt)['input_ids']





    pbar.close()

    return 0


if __name__ == '__main__':
    def rethrow(e):
        raise e
    run_and_exit(ArgsGenBertEmbs, main, 'Run Bert model embeddings inference for Wikipedia dataset.', exception_handler=rethrow)


