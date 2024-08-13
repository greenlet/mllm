import gzip
import itertools as it
from pathlib import Path
from typing import Optional, Callable, TextIO

import numpy as np
import pandas as pd
import torch

from mllm.tokenization.chunk_tokenizer import parse_out_subdir, ChunkTokenizer, split_doc_embs

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
        docid, url, title, body = l.rstrip('\n').split('\t')
        return MsmDoc(docid=docid, url=url, title=title, body=body)


def get_doc(fid: TextIO, offset: int) -> MsmDoc:
    fid.seek(offset)
    l = fid.readline().rstrip('\n')
    return MsmDoc.from_line(l)


class MsmDocsBatch:
    docs_chunks: list[list[np.ndarray]]
    qs_chunks: list[list[list[int]]]
    pad_tok: int
    emb_chunk_size: int
    docs_chunks_padded: np.ndarray
    qs_chunks_padded: np.ndarray
    docs_off_len: list[tuple[int, int]]
    qs_off_len: list[tuple[int, int]]
    docs_chunks_padded_tf: Optional[torch.Tensor] = None
    qs_chunks_padded_tf: Optional[torch.Tensor] = None
    device: Optional[torch.device] = None

    def __init__(self, docs_chunks: list[list[np.ndarray]], qs_chunks: list[list[list[int]]], pad_tok: int, emb_chunk_size: int,
                 device: Optional[torch.device] = None):
        self.docs_chunks = docs_chunks
        self.qs_chunks = qs_chunks
        self.pad_tok = pad_tok
        self.emb_chunk_size = emb_chunk_size
        self.device = device
        self.calc_np()

    def calc_np(self):
        assert len(self.docs_chunks) == len(self.qs_chunks), f'# of docs ({len(self.docs_chunks)}) != # of queries ({len(self.qs_chunks)})'

        max_chunks_sz = 0
        docs_off_len, docs_off = [], 0
        for chunk in self.docs_chunks:
            max_chunks_sz = max(max_chunks_sz, max(len(tokens) for tokens in chunk))
            n_chunk = len(chunk)
            docs_off_len.append((docs_off, n_chunk))
            docs_off += n_chunk

        qs_off_len, qs_off = [], 0
        for chunk in self.qs_chunks:
            max_chunks_sz = max(max_chunks_sz, max(len(tokens) for tokens in chunk))
            n_chunk = len(chunk)
            qs_off_len.append((qs_off, n_chunk))
            qs_off += n_chunk

        docs_chunks_padded = np.full((docs_off, max_chunks_sz), self.pad_tok, dtype=np.int32)
        for i_doc, doc_chunk in enumerate(self.docs_chunks):
            for i_tok, tokens in enumerate(doc_chunk):
                off = docs_off_len[i_doc][0]
                docs_chunks_padded[off + i_tok, :len(tokens)] = tokens

        qs_chunks_padded = np.full((qs_off, max_chunks_sz), self.pad_tok, dtype=np.int32)
        for i_query, query_chunk in enumerate(self.qs_chunks):
            for i_tok, tokens in enumerate(query_chunk):
                off = qs_off_len[i_query][0]
                qs_chunks_padded[off + i_tok, :len(tokens)] = tokens

        self.docs_chunks_padded = docs_chunks_padded
        self.qs_chunks_padded = qs_chunks_padded
        self.docs_off_len = docs_off_len
        self.qs_off_len = qs_off_len

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        res = torch.from_numpy(arr)
        if self.device is not None:
            res = res.to(self.device)
        return res

    def gen_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.docs_chunks_padded_tf is None:
            self.docs_chunks_padded_tf, self.qs_chunks_padded_tf = \
                map(self._to_tensor, (self.docs_chunks_padded, self.qs_chunks_padded))
        return self.docs_chunks_padded_tf, self.qs_chunks_padded_tf


class MsmDsLoader:
    ds_path: Path
    emb_chunk_size: int
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

    def __init__(self, ds_path: Path, emb_chunk_size: int, docs_batch_size: int, max_chunks_per_doc: int,
                 pad_tok: int, qbeg_tok: int, qend_tok: int, ch_tkz: ChunkTokenizer, device: Optional[torch.device] = None):
        self.ds_path = ds_path
        self.emb_chunk_size = emb_chunk_size
        self.docs_batch_size = docs_batch_size
        self.max_chunks_per_doc = max_chunks_per_doc
        self.pad_tok = pad_tok
        self.qbeg_tok = qbeg_tok
        self.qend_tok = qend_tok
        self.device = device
        assert ch_tkz.fixed_size and ch_tkz.dir_out is None, f'fixed_size = {ch_tkz.fixed_size}. dir_out = {ch_tkz.dir_out}'
        self.ch_tkz = ch_tkz

        # qs_train_fpath = self.ds_path / MSMARCO_DOCTRAIN_QUERIES_FNAME
        # qrels_train_fpath = self.ds_path / MSMARCO_DOCTRAIN_QRELS_FNAME
        # print(f'Loading {qs_train_fpath}')
        # self.df_qs_train = read_queries_df(qs_train_fpath)
        # print(f'Loading {qrels_train_fpath}')
        # self.df_qrels_train = read_qrels_df(qrels_train_fpath)
        # qs_val_fpath = self.ds_path / MSMARCO_DOCDEV_QUERIES_FNAME
        # qrels_val_fpath = self.ds_path / MSMARCO_DOCDEV_QRELS_FNAME
        # print(f'Loading {qs_val_fpath}')
        # self.df_qs_val = read_queries_df(qs_val_fpath)
        # print(f'Loading {qrels_val_fpath}')
        # self.df_qrels_val = read_qrels_df(qrels_val_fpath)
        # docs_fpath = self.ds_path / MSMARCO_DOCS_FNAME
        # self.fid_docs = open_fid_docs(docs_fpath)
        # lookup_fpath = self.ds_path / MSMARCO_DOCS_LOOKUP_FNAME
        # print(f'Loading {lookup_fpath}')
        # self.df_off = read_offsets_df(lookup_fpath)
        # self.n_qs_train = len(self.df_qrels_train)
        # self.n_qs_val = len(self.df_qrels_train)
        # self.qids_train = self.df_qrels_train.index.to_numpy().copy()
        # self.qids_val = self.df_qrels_val.index.to_numpy().copy()

        qs_train_fpath = self.ds_path / MSMARCO_DOCTRAIN_QUERIES_FNAME
        qrels_train_fpath = self.ds_path / MSMARCO_DOCTRAIN_QRELS_FNAME
        print(f'Loading {qs_train_fpath}')
        df_qs_1 = read_queries_df(qs_train_fpath)
        print(f'Loading {qrels_train_fpath}')
        df_qrels_1 = read_qrels_df(qrels_train_fpath)
        qs_val_fpath = self.ds_path / MSMARCO_DOCDEV_QUERIES_FNAME
        qrels_val_fpath = self.ds_path / MSMARCO_DOCDEV_QRELS_FNAME
        print(f'Loading {qs_val_fpath}')
        df_qs_2 = read_queries_df(qs_val_fpath)
        print(f'Loading {qrels_val_fpath}')
        df_qrels_2 = read_qrels_df(qrels_val_fpath)
        docs_fpath = self.ds_path / MSMARCO_DOCS_FNAME
        fid_docs = open_fid_docs(docs_fpath)
        lookup_fpath = self.ds_path / MSMARCO_DOCS_LOOKUP_FNAME
        print(f'Loading {lookup_fpath}')
        df_off = read_offsets_df(lookup_fpath)

        df_qs = pd.concat([df_qs_1, df_qs_2], axis=0)
        df_qrels = pd.concat([df_qrels_1, df_qrels_2], axis=0)
        qids = df_qrels.index.to_numpy().copy()
        np.random.shuffle(qids)
        df_qrels = df_qrels.loc[qids]
        n_qs = len(df_qrels)
        self.n_qs_train = int(n_qs * 0.9)
        self.n_qs_val = n_qs - self.n_qs_train
        self.qids_train = qids[:self.n_qs_train].copy()
        self.qids_val = qids[self.n_qs_train:].copy()
        self.df_qs_train = df_qs.loc[self.qids_train]
        self.df_qs_val = df_qs.loc[self.qids_val]
        self.df_qrels_train = df_qrels.loc[self.qids_train]
        self.df_qrels_val = df_qrels.loc[self.qids_val]

        self.fid_docs = fid_docs
        self.df_off = df_off

    def tokenize_query(self, query: str) -> list[list[int]]:
        tokens = self.ch_tkz.tokenizer(query)['input_ids']
        # tokens = [self.qbeg_tok, *tokens, self.qend_tok]
        off = split_doc_embs(len(tokens), self.emb_chunk_size, fixed_size=True)
        res = []
        for i in range(len(off) - 1):
            res.append(tokens[off[i]:off[i + 1]])
        return res

    def get_batch(self, ind: int, train: bool) -> MsmDocsBatch:
        qids, df_qrels, df_qs = self.qids_train, self.df_qrels_train, self.df_qs_train
        if train:
            qids, df_qrels, df_qs = self.qids_val, self.df_qrels_val, self.df_qs_val
        i1 = ind * self.docs_batch_size
        i2 = min(i1 + self.docs_batch_size, len(qids))
        i1 = i2 - self.docs_batch_size
        assert 0 <= i1 < i2 <= len(qids), f'i1 = {i1}. i2 = {i2}. len(qids) = {len(qids)}'
        qids = qids[i1:i2]
        df_qrels = df_qrels.loc[qids]

        docs_chunks, qs_chunks = [], []
        for _, row in df_qrels.iterrows():
            docidn = row['docidn']
            off = self.df_off.loc[docidn]['off_tsv']
            doc = get_doc(self.fid_docs, off)
            body = f'{doc.url} {doc.body}' if doc.body else doc.url
            doc_chunks = self.ch_tkz.process_doc(doc.docidn, {'title': doc.title, 'text': body})
            if len(doc_chunks) > self.max_chunks_per_doc:
                i = np.random.randint(len(doc_chunks) - self.max_chunks_per_doc + 1)
                doc_chunks = doc_chunks[i:i + self.max_chunks_per_doc]
            qid = row.name
            query = df_qs.loc[qid]['query']
            query_chunks = self.tokenize_query(query)
            docs_chunks.append([ch.tokens for ch in doc_chunks])
            qs_chunks.append(query_chunks)

        return MsmDocsBatch(
            docs_chunks=docs_chunks, qs_chunks=qs_chunks,
            pad_tok=self.pad_tok, emb_chunk_size=self.emb_chunk_size, device=self.device,
        )

    def shuffle(self, train: bool):
        qids = self.qids_train if train else self.qids_val
        np.random.shuffle(qids)

    def close(self):
        self.fid_docs.close()
        self.fid_docs = None


