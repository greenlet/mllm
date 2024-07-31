import gzip
from io import TextIOWrapper
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from mllm.tokenization.chunk_tokenizer import parse_out_subdir

MSMARCO_DOCTRAIN_QUERIES_FNAME = 'msmarco-doctrain-queries.tsv.gz'
MSMARCO_DOCTRAIN_QRELS_FNAME = 'msmarco-doctrain-qrels.tsv.gz'
MSMARCO_DOCTRAIN_TOP100_FNAME = 'msmarco-doctrain-top100.gz'
MSMARCO_DOCDEV_QUERIES_FNAME = 'msmarco-docdev-queries.tsv.gz'
MSMARCO_DOCDEV_QRELS_FNAME = 'msmarco-docdev-qrels.tsv.gz'
MSMARCO_DOCDEV_TOP100_FNAME = 'msmarco-docdev-top100.gz'
MSMARCO_DOCS_FNAME = 'msmarco-docs.tsv'
MSMARCO_DOCS_LOOKUP_FNAME = 'msmarco-docs-lookup.tsv.gz'


def docid_to_num(docid: str) -> int:
    return int(docid[1:])


def read_queries_df(queries_fpath: Path) -> pd.DataFrame:
    with gzip.open(queries_fpath, 'rt', encoding='utf8') as f:
        df = pd.read_csv(f, sep='\t', header = None, names=('topicid', 'query'))
        df.set_index('topicid', inplace=True)
    return df


def read_offsets_df(lookup_fpath: Path) -> pd.DataFrame:
    with gzip.open(lookup_fpath, 'rt', encoding='utf8') as f:
        df = pd.read_csv(f, sep='\t', header=None, names=('docid', 'off_trec', 'off_tsv'), usecols=('docid', 'off_tsv'))
        df['docidn'] = df['docid'].apply(docid_to_num)
        df.set_index('docidn', inplace=True)
    return df


def read_qrels_df(qrels_fpath: Path) -> pd.DataFrame:
    with gzip.open(qrels_fpath, 'rt', encoding='utf8') as f:
        df = pd.read_csv(f, sep=' ', header=None, names=('topicid', 'x', 'docid', 'rel'), usecols=('topicid', 'docid', 'rel'))
        df['docidn'] = df['docid'].apply(docid_to_num)
        df.set_index('topicid', inplace=True)
    assert len(df.index.unique()) == len(df)
    assert (df['rel'] == 1).sum() == len(df)
    return df


def read_top_df(top_fpath: Path) -> pd.DataFrame:
    with gzip.open(top_fpath, 'rt', encoding='utf8') as f:
        df = pd.read_csv(f, sep=' ', header=None, names=('topicid', 'x', 'docid', 'rank', 'score', 'runstring'),
                         usecols=('topicid', 'docid', 'rank', 'score'))
        df.set_index(['topicid', 'docid'], inplace=True)
    return df


def cut(s: str, sz: int) -> str:
    if len(s) <= sz:
        return s
    return f'{s[:sz]}...'


class MsmDoc:
    docid: str
    url: str
    title: str
    body: str

    def __init__(self, docid: str, url: str, title: str, body: str) -> None:
        self.docid = docid
        self.url = url
        self.title = title
        self.body = body

    def __str__(self) -> str:
        return f'Id: {self.docid}. Title: {cut(self.title, 50)}. Body: {cut(self.body, 100)}. Url: {self.url}'

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_line(l: str) -> 'MsmDoc':
        docid, url, title, body = l.rstrip().split('\t')
        return MsmDoc(docid=docid, url=url, title=title, body=body)


def get_doc(fid: TextIOWrapper, offset: int) -> MsmDoc:
    fid.seek(offset)
    l = fid.readline().rstrip()
    return MsmDoc.from_line(l)


class DsLoader:
    ds_path: Path
    emb_chunk_size: int
    fixed_size: bool
    docs_batch_size: int
    max_chunks_per_doc: int
    pad_tok: int
    qbeg_tok: int
    qend_tok: int
    df_qs_train: pd.DataFrame
    df_qrels_train: pd.DataFrame
    df_qs_val: pd.DataFrame
    df_qrels_val: pd.DataFrame
    n_qs_train: int
    n_qs_val: int
    qids_train: np.ndarray
    qids_val: np.ndarray
    device: Optional[torch.device] = None

    def __init__(self, ds_path: Path, docs_batch_size: int, max_chunks_per_doc: int,
                 pad_tok: int, qbeg_tok: int, qend_tok: int):
        self.ds_path = ds_path
        self.emb_chunk_size, self.fixed_size = parse_out_subdir(ds_path.name)
        self.docs_batch_size = docs_batch_size
        self.max_chunks_per_doc = max_chunks_per_doc
        self.pad_tok = pad_tok
        self.qbeg_tok = qbeg_tok
        self.qend_tok = qend_tok

        qs_train_fpath = self.ds_path / MSMARCO_DOCTRAIN_QUERIES_FNAME
        qrels_train_fpath = self.ds_path / MSMARCO_DOCTRAIN_QRELS_FNAME
        print(f'Loading {qs_train_fpath}')
        self.df_qs_train = read_queries_df(qs_train_fpath)
        print(f'Loading {qrels_train_fpath}')
        self.df_qrels_train = read_queries_df(qrels_train_fpath)
        qs_val_fpath = self.ds_path / MSMARCO_DOCDEV_QUERIES_FNAME
        qrels_val_fpath = self.ds_path / MSMARCO_DOCDEV_QRELS_FNAME
        print(f'Loading {qs_val_fpath}')
        self.df_qs_val = read_queries_df(qs_val_fpath)
        print(f'Loading {qrels_val_fpath}')
        self.df_qrels_val = read_queries_df(qrels_val_fpath)
        self.n_qs_train = len(self.df_qrels_train)
        self.n_qs_val = len(self.df_qrels_train)
        self.qids_train = self.df_qrels_train.index.to_numpy().copy()
        self.qids_val = self.df_qrels_val.index.to_numpy().copy()


