from pathlib import Path

import numpy as np
from datasets import load_dataset
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit

from mllm.tokenization.chunk_tokenizer import ChunkTokenizer, gen_add_doc_tokens
from transformers import GPT2Tokenizer


class ArgsPreproc(BaseModel):
    ds_path: Path = Field(
        None,
        required=False,
        description='Path to a dataset loaded within a `datasets` module.',
        cli=('--ds-path',),
    )
    out_path: Path = Field(
        ...,
        required=True,
        description='Path to tokenized data.',
        cli=('--out-path',),
    )


def main(args: ArgsPreproc) -> int:
    print(args)
    ds_cache_dir, ds_path, ds_name = args.ds_path.parent.parent, args.ds_path.parent.name, args.ds_path.name
    ds = load_dataset(path=ds_path, name=ds_name, beam_runner='DirectRunner', cache_dir=str(ds_cache_dir))
    ds_train = ds['train']
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    all_tokens = gen_add_doc_tokens(tokenizer)


    txt = 'Привет всем!<|endoftext|> Hola! àáâäæãåā <|pad|> <|doc_begin|>'
    print(txt)
    tokens = tokenizer(txt)
    print(tokens['input_ids'], len(tokens['input_ids']))
    txt2 = tokenizer.decode(tokens['input_ids'])
    print(txt2)

    n_ds = len(ds_train)
    print(f'Dataset size: {n_ds}')
    n_emb_tokens = 100
    dir_out = ...
    docs_write_num = 100
    chtkz = ChunkTokenizer(tokens=all_tokens, tokenizer=tokenizer, n_emb_tokens=n_emb_tokens)
    for i in range(n_ds):
        art = ds_train[i]
        title, text = art['title'], art['text']
        title_tok, text_tok = tokenizer(title), tokenizer(text)
        doc_tok = np.concatenate
        break

    return 0


if __name__ == '__main__':
    run_and_exit(ArgsPreproc, main, 'Tokenize and split in chunks text dataset.')

