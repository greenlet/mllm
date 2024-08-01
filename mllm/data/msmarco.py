import gzip
from pathlib import Path
from typing import Optional, Callable, TextIO

import numpy as np
import pandas as pd
import torch

from mllm.data.common import DocsBatch
from mllm.tokenization.chunk_tokenizer import parse_out_subdir, ChunkTokenizer

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


def open_fid_docs(docs_fpath: Path) -> TextIO:
    fid_docs = open(docs_fpath, 'r', encoding='utf-8')
    return fid_docs


class MsmDoc:
    docid: str
    docidn: int
    url: str
    title: str
    body: str

    def __init__(self, docid: str, url: str, title: str, body: str) -> None:
        self.docid = docid
        self.docidn = docid_to_num(docid)
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


def get_doc(fid: TextIO, offset: int) -> MsmDoc:
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
    ch_tkz: ChunkTokenizer
    df_qs_train: pd.DataFrame
    df_qrels_train: pd.DataFrame
    df_qs_val: pd.DataFrame
    df_qrels_val: pd.DataFrame
    df_off: pd.DataFrame
    n_qs_train: int
    n_qs_val: int
    qids_train: np.ndarray
    qids_val: np.ndarray
    device: Optional[torch.device] = None
    fid_docs: Optional[TextIO] = None

    def __init__(self, ds_path: Path, docs_batch_size: int, max_chunks_per_doc: int,
                 pad_tok: int, qbeg_tok: int, qend_tok: int, ch_tkz: ChunkTokenizer):
        self.ds_path = ds_path
        self.emb_chunk_size, self.fixed_size = parse_out_subdir(ds_path.name)
        self.docs_batch_size = docs_batch_size
        self.max_chunks_per_doc = max_chunks_per_doc
        self.pad_tok = pad_tok
        self.qbeg_tok = qbeg_tok
        self.qend_tok = qend_tok
        assert ch_tkz.fixed_size and ch_tkz.dir_out is None, f'fixed_size = {ch_tkz.fixed_size}. dir_out = {ch_tkz.dir_out}'
        self.ch_tkz = ch_tkz

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
        docs_fpath = self.ds_path / MSMARCO_DOCS_FNAME
        lookup_fpath = self.ds_path / MSMARCO_DOCS_LOOKUP_FNAME
        print(f'Loading {lookup_fpath}')
        self.df_off = read_offsets_df(lookup_fpath)
        self.n_qs_train = len(self.df_qrels_train)
        self.n_qs_val = len(self.df_qrels_train)
        self.qids_train = self.df_qrels_train.index.to_numpy().copy()
        self.qids_val = self.df_qrels_val.index.to_numpy().copy()

    def _tokenize_query(self, query: str) -> np.ndarray:
        pass
    def get_batch(self, ind: int, train: bool, target_augmenter: Optional[Callable] = None) -> DocsBatch:
        qids, df_qrels = self.qids_train, self.df_qs_train
        if train:
            qids, df_qrels = self.qids_val, self.df_qs_val
        i1 = ind * self.docs_batch_size
        i2 = min(i1 + self.docs_batch_size, len(qids))
        i1 = i2 - self.docs_batch_size
        assert 0 <= i1 < i2 < len(qids)
        qids = qids[i1:i2]
        df_qrels = df_qrels.loc[qids]

        for _, row in df_qrels.iterrows():
            docidn = row['docidn']
            off = self.df_off.loc[docidn]['off_tsv']
            doc = get_doc(self.fid_docs, off)
            doc_chunks = self.ch_tkz.process_doc(doc.docidn, {'title': doc.title, 'text': doc.body})
            # query_chunks = self.ch_tkz.process_doc()

        return DocsBatch(
            docs_chunks=docs_chunks, target_doc_id=target_docid, target_tokens=target_tokens,
            pad_tok=self.pad_tok, qbeg_tok=self.qbeg_tok, qend_tok=self.qend_tok,
            emb_chunk_size=self.emb_chunk_size, fixed_size=self.fixed_size, device=self.device,
        )

    def shuffle(self, train: bool):
        qids = self.qids_train if train else self.qids_val
        np.random.shuffle(qids)

    def close(self):
        self.fid_docs.close()
        self.fid_docs = None