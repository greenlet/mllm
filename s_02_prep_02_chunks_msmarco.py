import shutil
from pathlib import Path

import numpy as np
from datasets import load_dataset
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from tqdm import trange

from mllm.data.dsmsmarco import MSMARCO_DOCS_FNAME, MsmDoc
from mllm.tokenization.chunk_tokenizer import ChunkTokenizer, gen_out_subdir, gen_all_tokens
from transformers import GPT2Tokenizer


class ArgsPreproc(BaseModel):
    ds_path: Path = Field(
        None,
        required=False,
        description='Path to Msmarco dataset.',
        cli=('--ds-path',),
    )
    emb_chunk_size: int = Field(
        100,
        required=False,
        description='Number of embeddings in a chunk',
        cli=('--emb-chunk-size',),
    )
    chunk_fixed_size: bool = Field(
        False,
        required=False,
        description='If set, each chunk size will be exactly EMB_CHUNK_SIZE. Otherwise,'
                    'chunks sizes will be around EMB_CHUNK_SIZE and without padding tokens.',
        cli=('--chunk-fixed-size',)
    )
    max_docs: int = Field(
        0,
        required=False,
        description='Maximum documents to split in chunks. If MAX_DOCS <= 0 all documents will be processed',
        cli=('--max-docs',)
    )
    out_path: Path = Field(
        ...,
        required=True,
        description='Path to tokenized data.',
        cli=('--out-path',),
    )


def main(args: ArgsPreproc) -> int:
    print(args)

    dir_out = ''
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', model_max_length=100000)
    all_tokens = gen_all_tokens(tokenizer)
    ch_tkz = ChunkTokenizer(
        tokens=all_tokens, tokenizer=tokenizer, n_emb_tokens=args.emb_chunk_size,
        fixed_size=args.chunk_fixed_size, dir_out=dir_out, docs_write_num=100,
    )
    docs_fpath = args.ds_path / MSMARCO_DOCS_FNAME
    with open(docs_fpath, 'r', encoding='utf-8') as f:
        while True:
            l = f.readline().strip()
            if not l: break
            doc = MsmDoc.from_line(l)

    return 0


if __name__ == '__main__':
    def rethrow(e):
        raise e
    run_and_exit(ArgsPreproc, main, 'Tokenize Msmarco dataset and split in chunks text dataset.', exception_handler=rethrow)

