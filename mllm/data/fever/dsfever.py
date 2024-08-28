from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from mllm.data.dsqrels import DsQrels, DocsFile, DsQrelsId
from mllm.tokenization.chunk_tokenizer import ChunkTokenizer
from mllm.utils.utils import read_tsv

SPLITS = 'train', 'dev', 'test'
QS_FNAME_PAT = 'queries_{0}.tsv'
QRELS_FNAME_PAT = 'qrels_{0}.tsv'
DOCS_OFF_FNAME = 'docs_offsets.tsv'
DOCS_FNAME = 'docs.tsv'


def get_qs_fname(split: str) -> str:
    assert split in SPLITS
    return QS_FNAME_PAT.format(split)


def get_qrels_fname(split: str) -> str:
    assert split in SPLITS
    return QRELS_FNAME_PAT.format(split)


def load_dsqrels_fever(
        ds_path: Path, ch_tkz: ChunkTokenizer, max_chunks_per_doc: int, emb_chunk_size: int,
        device: Optional[torch.device] = None) -> DsQrels:
    qs_splits, qrels_splits = [], []
    for split in SPLITS:
        qs_fpath = ds_path / get_qs_fname(split)
        qrels_fpath = ds_path / get_qrels_fname(split)
        df_qs = read_tsv(qs_fpath)
        df_qrels = read_tsv(qrels_fpath)
        qs_splits.append(df_qs)
        qrels_splits.append(df_qrels)
    df_qs = pd.concat(qs_splits, axis=0)
    df_qrels = pd.concat(qrels_splits, axis=0)
    docs_off_fpath, docs_fpath = ds_path / DOCS_OFF_FNAME, ds_path / DOCS_FNAME
    df_off = read_tsv(docs_off_fpath)
    docs_file = DocsFile(docs_fpath)

    df_qs.rename({'queryid': 'qid'}, inplace=True)
    df_qrels.rename({'queryid': 'qid', 'docid': 'did'}, inplace=True)
    df_off.rename({'docid': 'did'}, inplace=True)
    ds_id = DsQrelsId.Fever
    return DsQrels(
        ch_tkz=ch_tkz, ds_ids=[ds_id], dfs_qs=[df_qs], dfs_qrels=[df_qrels], dfs_off=[df_off], docs_files={ds_id: docs_file},
        max_chunks_per_doc=max_chunks_per_doc, emb_chunk_size=emb_chunk_size, device=device,
    )


