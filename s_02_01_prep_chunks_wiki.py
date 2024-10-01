import shutil
from pathlib import Path

from datasets import load_dataset
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from tqdm import trange

from mllm.tokenization.chunk_tokenizer import ChunkTokenizer, gen_out_subdir, gen_all_tokens
from transformers import GPT2Tokenizer


class ArgsPreproc(BaseModel):
    ds_path: Path = Field(
        None,
        required=False,
        description='Path to a dataset loaded within a `datasets` module.',
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
    ds_cache_dir, ds_path, ds_name = args.ds_path.parent.parent, args.ds_path.parent.name, args.ds_path.name
    ds = load_dataset(path=ds_path, name=ds_name, beam_runner='DirectRunner', cache_dir=str(ds_cache_dir))
    ds_train = ds['train']
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', model_max_length=100000)
    all_tokens = gen_all_tokens(tokenizer)

    subdir = gen_out_subdir(args.emb_chunk_size, args.chunk_fixed_size)
    dir_out = args.out_path / subdir
    shutil.rmtree(dir_out, ignore_errors=True)
    ch_tkz = ChunkTokenizer(
        tokens=all_tokens, tokenizer=tokenizer, n_emb_tokens=args.emb_chunk_size,
        fixed_size=args.chunk_fixed_size, dir_out=dir_out, docs_write_num=100,
    )

    n_ds = len(ds_train)
    print(f'Dataset size: {n_ds}')
    n_docs = min(n_ds, args.max_docs) if args.max_docs > 0 else n_ds
    print(f'Documents to process: {n_docs}')

    pbar = trange(n_docs, desc=f'ChunkGen', unit='doc')
    for i in pbar:
        doc = ds_train[i]
        ch_tkz.process_doc(i, doc)
    ch_tkz.write_data()
    pbar.close()

    return 0


if __name__ == '__main__':
    def rethrow(e):
        raise e
    run_and_exit(ArgsPreproc, main, 'Tokenize and split in chunks text dataset.', exception_handler=rethrow)

