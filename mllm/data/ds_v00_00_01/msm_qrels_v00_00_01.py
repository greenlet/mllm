from pathlib import Path
from typing import Optional, TextIO

import numpy as np
import pandas as pd
import torch

from mllm.data.dsqrels import QrelsBatch, DsQrelsView, DsQrels
from mllm.data.msmarco.dsmsmarco import MSMARCO_DOCTRAIN_QUERIES_FNAME, MSMARCO_DOCTRAIN_QRELS_FNAME, read_queries_df, \
    read_qrels_df, MSMARCO_DOCDEV_QUERIES_FNAME, MSMARCO_DOCDEV_QRELS_FNAME, MSMARCO_DOCS_FNAME, open_fid_docs, \
    MSMARCO_DOCS_LOOKUP_FNAME, read_offsets_df
from mllm.tokenization.chunk_tokenizer import ChunkTokenizer

VERSION = '0.0.1'
DESCRIPTION = '''
MSMARCO dataset
'''


class DsMsmQrels(DsQrels):
    ds_id = 'msmarco'
    ds_path: Path
    emb_chunk_size: int
    docs_batch_size: int
    max_chunks_per_doc: int
    pad_tok: int
    ch_tkz: ChunkTokenizer
    df_qs_train: pd.DataFrame
    df_qrels_train: pd.DataFrame
    df_qs_dev: pd.DataFrame
    df_qrels_dev: pd.DataFrame
    df_off: pd.DataFrame
    fid_docs: Optional[TextIO] = None

    def __init__(self):
        qs_train_fpath = self.ds_path / MSMARCO_DOCTRAIN_QUERIES_FNAME
        qrels_train_fpath = self.ds_path / MSMARCO_DOCTRAIN_QRELS_FNAME
        print(f'Loading {qs_train_fpath}')
        self.df_qs_train = read_queries_df(qs_train_fpath)
        print(f'Loading {qrels_train_fpath}')
        self.df_qrels_train = read_qrels_df(qrels_train_fpath)
        qs_dev_fpath = self.ds_path / MSMARCO_DOCDEV_QUERIES_FNAME
        qrels_dev_fpath = self.ds_path / MSMARCO_DOCDEV_QRELS_FNAME
        print(f'Loading {qs_dev_fpath}')
        self.df_qs_dev = read_queries_df(qs_dev_fpath)
        print(f'Loading {qrels_dev_fpath}')
        self.df_qrels_dev = read_qrels_df(qrels_dev_fpath)
        docs_fpath = self.ds_path / MSMARCO_DOCS_FNAME
        self.fid_docs = open_fid_docs(docs_fpath)
        lookup_fpath = self.ds_path / MSMARCO_DOCS_LOOKUP_FNAME
        print(f'Loading {lookup_fpath}')
        self.df_off = read_offsets_df(lookup_fpath)


    def close(self):
        self.fid_docs.close()
        self.fid_docs = None


def test_qsmsmqrels():
    pass


if __name__ == '__main__':
    ds = DsMsmQrels()
    print(ds.ds_id)
    print(ds.qs_ids)

