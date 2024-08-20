from pathlib import Path

import pandas as pd

from mllm.data.dsqrels import DsQrels
from mllm.utils.utils import read_tsv

SPLITS = 'train', 'dev', 'test'
QS_FNAME_PAT = 'queries_{0}.tsv'
QRELS_FNAME_PAT = 'qrels_{0}.tsv'


def get_qs_fname(split: str) -> str:
    assert split in SPLITS
    return QS_FNAME_PAT.format(split)


def get_qrels_fname(split: str) -> str:
    assert split in SPLITS
    return QRELS_FNAME_PAT.format(split)



class DsFever(DsQrels):
    ds_path: Path

    def __init__(self, ds_path: Path):
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

        super().__init__()
