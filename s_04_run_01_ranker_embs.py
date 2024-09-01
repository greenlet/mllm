import shutil
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic.main import BaseModel
from pydantic_cli import run_and_exit
import torch
import torch.utils.tensorboard as tb
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange

from mllm.data.dsqrels import DsQrels
from mllm.data.fever.dsfever import load_dsqrels_fever
from mllm.data.msmarco.dsmsmarco import MsmDsLoader, load_dsqrels_msmarco
from mllm.model.mllm_ranker import MllmRanker, RankProbLoss
from mllm.model.mllm_encdec import MllmEncdec
from mllm.exp.cfg import create_mllm_encdec_cfg, create_mllm_ranker_cfg
from mllm.tokenization.chunk_tokenizer import gen_all_tokens, ChunkTokenizer
from mllm.exp.args import ArgsTokensChunksTrain
from mllm.train.utils import find_create_train_path
from transformers import GPT2Tokenizer


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
    docs_batch_size: int = Field(
        3,
        required=False,
        description='Documents batch size for inference.',
        cli=('--docs-batch-size',),
    )
    device: str = Field(
        'cpu',
        required=False,
        description='Device to run inference on. Can have values: "cpu", "cuda"',
        cli=('--device',)
    )


def main(args: ArgsRunRankerEmbs) -> int:
    print(args)

    assert args.ds_dir_paths, '--ds-dir-paths is expected to list at least one Qrels datsaset'

    device = torch.device(args.device)

    ds_names = '-'.join([dpath.name for dpath in args.ds_dir_paths])
    train_path = args.train_root_path / args.train_subdir
    print(f'train_path: {train_path}')

    last_checkpoint_path = train_path / 'last.pth'
    checkpoint = torch.load(last_checkpoint_path, map_location=device)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', model_max_length=10000)
    tok_dict = gen_all_tokens(tokenizer)
    ch_tkz = ChunkTokenizer(tok_dict, tokenizer, n_emb_tokens=args.emb_chunk_size, fixed_size=True)
    pad_tok, qbeg_tok, qend_tok = tok_dict['pad'].ind, tok_dict['query_begin'].ind, tok_dict['query_end'].ind

    dss = []
    for ds_path in args.ds_dir_paths:
        if 'fever' in ds_path.name:
            load_fn = load_dsqrels_fever
        elif 'msmarco' in ds_path.name:
            load_fn = load_dsqrels_msmarco
        else:
            raise Exception(f'Unknown dataset: {ds_path}')
        ds = load_fn(ds_path=ds_path, ch_tkz=ch_tkz, max_chunks_per_doc=100, emb_chunk_size=args.emb_chunk_size, device=device)
        dss.append(ds)

    print('Join datasets:')
    for ds in dss:
        assert len(ds.ds_ids) == 1
        print(f'   {ds}')
    ds = DsQrels.join(dss)
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
    print(model)
    return 0


if __name__ == '__main__':
    def rethrow(e):
        raise e
    run_and_exit(ArgsRunRankerEmbs, main, 'Run Mllm Ranking model inference to form chunks for the next level.', exception_handler=rethrow)


