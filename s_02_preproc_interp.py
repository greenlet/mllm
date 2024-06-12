import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from transformers import GPT2Tokenizer, PreTrainedTokenizer
from tqdm import trange


def gen_doc_special_tokens_names(doc_token_names: list[str]) -> list[str]:
    token_names = []
    prefix = 'doc'
    for tname in doc_token_names:
        tname = prefix if not tname else f'{prefix}_{tname}'
        token_names.extend((f'{tname}_begin', f'{tname}_end'))
    return token_names


SPECIAL_TOKENS_NAMES = gen_doc_special_tokens_names(['', 'id', 'offset', 'title', 'body'])
SPECIAL_TOKENS_NAMES += ['pad']
SPECIAL_TOKENS = {f'{tname}_token': {'view': f'<|{tname}|>'} for tname in SPECIAL_TOKENS_NAMES}


def add_special_tokens(tokenizer: PreTrainedTokenizer, tokens: dict[str, dict[str, any]]) -> dict[str, dict[str, any]]:
    tokens = {tname: token['view'] for tname, token in tokens.items()}
    tokenizer.add_special_tokens(tokens)


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



    print(tokenizer.pad_token)
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    print(tokenizer)
    print(tokenizer.pad_token)
    print(len(tokenizer))

    txt = 'Привет всем!<|endoftext|> Hola! àáâäæãåā <|pad|>'
    print(txt)
    tokens = tokenizer(txt)
    print(tokens['input_ids'], len(tokens['input_ids']))
    txt2 = tokenizer.decode(tokens['input_ids'])
    print(txt2)

    return 0


if __name__ == '__main__':
    run_and_exit(ArgsPreproc, main, 'Tokenize text dataset.')

