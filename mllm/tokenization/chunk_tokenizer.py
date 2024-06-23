from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from transformers import PreTrainedTokenizer


class CustomToken:
    name: str
    repr: str
    ind: int = 0

    def __init__(self, name: str):
        self.name = name
        self.repr = f'<|{name}|>'

    def set_ind(self, ind: int):
        self.ind = ind

    def __str__(self) -> str:
        return f'{self.__class__.__name__}. Name: {self.name}. Repr: {self.repr}. Ind: {self.ind}'

    def __repr__(self) -> str:
        return str(self)


def prefix(pref: str, s: str) -> str:
    if not pref:
        return s
    if not s:
        return pref
    return f'{pref}_{s}'


def gen_doc_tokens_names(tokens_basenames: list[str]) -> list[str]:
    res = []
    for tname in tokens_basenames:
        tname = prefix('doc', tname)
        res.extend((f'{tname}_begin', f'{tname}_end'))
    return res


def gen_tokens(tokens_names: list[str]) -> dict[str, CustomToken]:
    res = {}
    for tname in tokens_names:
        token = CustomToken(tname)
        res[token.name] = token
    return res


def gen_add_doc_tokens(tokenizer: PreTrainedTokenizer) -> dict[str, CustomToken]:
    doc_tokens_names = gen_doc_tokens_names(['', 'id', 'offset', 'title', 'body'])
    special_tokens_names = ['pad']
    doc_tokens = gen_tokens(doc_tokens_names)
    special_tokens = gen_tokens(special_tokens_names)
    for t in doc_tokens.values():
        tokenizer.add_tokens(t.repr)
        t.set_ind(len(tokenizer) - 1)
    for t in special_tokens.values():
        tokenizer.add_special_tokens({f'{t.name}_token': t.repr})
        t.set_ind(len(tokenizer) - 1)
    all_tokens = {**doc_tokens, **special_tokens}
    return all_tokens


class ChunkTokenizer:
    class TokChunk:
        docid: int
        tokens: np.ndarray
        offset: int
        docid_tok_num: int
        offset_tok_num: int
        title_tok_num: int
        body_tok_num: int

        def __init__(self, docid: int, tokens: np.ndarray, offset: int, docid_tok_num: int,
                offset_tok_num: int, title_tok_num: int, body_tok_num: int):
            self.docid = docid
            self.tokens = tokens
            self.offset = offset
            self.docid_tok_num = docid_tok_num
            self.offset_tok_num = offset_tok_num
            self.title_tok_num = title_tok_num
            self.body_tok_num = body_tok_num

    doc_beg_tok: int
    doc_end_tok: int
    id_beg_tok: int
    id_end_tok: int
    offset_beg_tok: int
    offset_end_tok: int
    title_beg_tok: int
    title_end_tok: int
    body_beg_tok: int
    body_end_tok: int
    pad_tok: int
    tokenizer: PreTrainedTokenizer
    n_emb_tokens: int
    fixed_size: bool
    dir_out: Optional[Path]
    docs_write_num: int
    docs_processed_num: int
    chunks: list[TokChunk]

    def __init__(self, tokens: dict[str, CustomToken], tokenizer: PreTrainedTokenizer, n_emb_tokens: int, fixed_size: bool,
                 dir_out: Optional[Path] = None, docs_write_num: int = 0):
        self.doc_beg_tok, self.doc_end_tok = tokens['doc_begin'].ind, tokens['doc_end'].ind
        self.id_beg_tok, self.id_end_tok = tokens['doc_id_begin'].ind, tokens['doc_id_end'].ind
        self.offset_beg_tok, self.offset_end_tok = tokens['doc_offset_begin'].ind, tokens['doc_offset_end'].ind
        self.title_beg_tok, self.title_end_tok = tokens['doc_title_begin'].ind, tokens['doc_title_end'].ind
        self.body_beg_tok, self.body_end_tok = tokens['doc_body_begin'].ind, tokens['doc_body_end'].ind
        self.pad_tok = tokens['pad'].ind
        self.tokenizer = tokenizer
        self.n_emb_tokens = n_emb_tokens
        self.fixed_size = fixed_size
        self.dir_out = dir_out
        self.docs_write_num = docs_write_num
        self.docs_processed_num = 0
        self.chunks = []

    def _write_data_if_needed(self):
        if self.docs_write_num <= 0 or self.dir_out is None or self.docs_processed_num < self.docs_write_num:
            return
        keys = 'docid', 'offset', 'tok_num', 'docid_tok_num', \
            'offset_tok_num', 'title_tok_num', 'body_tok_num'
        self.dir_out.mkdir(parents=True, exist_ok=True)
        n_chunks = len(self.chunks)
        data = {k: [0] * n_chunks for k in keys}
        tokens = [None] * n_chunks
        for i, chunk in enumerate(self.chunks):
            data['docid'][i] = chunk.docid
            data['offset'][i] = chunk.offset
            data['tok_num'][i] = len(chunk.tokens)
            data['docid_tok_num'][i] = chunk.docid_tok_num
            data['offset_tok_num'][i] = chunk.offset_tok_num
            data['title_tok_num'][i] = chunk.title_tok_num
            data['body_tok_num'][i] = chunk.body_tok_num

            tokens[i] = chunk.tokens

        doc_id_min = doc_id_max = self.chunks[0].docid, self.chunks[-1].docid
        fname_base = f'docs_{doc_id_min:07d}-{doc_id_max:07d}'
        fname_np, fname_csv = f'{fname_base}.np', f'{fname_base}.csv'
        fpath_np, fpath_csv = self.dir_out / fname_np, self.dir_out / fname_csv

        tokens = np.array(tokens, dtype=np.int32)
        tokens.tofile(fpath_np)

        df = pd.DataFrame(data)
        df.to_csv(fpath_csv)

        self.chunks = []
        self.docs_processed_num = 0

    def _split_fixed(self, docid: int, docid_tokens: list[int], title_tokens: list[int], body_tokens: list[int]) -> list[TokChunk]:
        n_docid = len(docid_tokens)
        n_title, n_body = len(title_tokens), len(body_tokens)
        n_doc = n_title + n_body
        ind, off = 0, 0
        chunks = []
        doc_end_written = False
        while not doc_end_written:
            offset = off
            tokens = np.full(self.n_emb_tokens, self.pad_tok, dtype=np.int32)
            i = 0
            tokens[i] = self.doc_beg_tok
            i += 1
            tokens[i:i + n_docid] = docid_tokens
            i += n_docid

            off_tokens = self.tokenizer(str(offset))['input_ids']
            off_tokens = [self.offset_beg_tok, *off_tokens, self.offset_end_tok]
            n_off = len(off_tokens)
            tokens[i:i + n_off] = off_tokens
            i += n_off

            n_rest = self.n_emb_tokens - i
            assert n_rest > 1, f'Doc tokens total: {self.n_emb_tokens}. Tokens occupied: {i}'

            title_tok_num = 0
            if off < n_title:
                off_cur = off
                n_cur = min(n_rest, n_title - off_cur)
                tokens[i:i + n_cur] = title_tokens[off_cur:off_cur + n_cur]
                off += n_cur
                n_rest -= n_cur
                i += n_cur
                title_tok_num = n_cur

            body_tok_num = 0
            if off >= n_title and n_rest > 0:
                off_cur = off - n_title
                n_cur = min(n_rest, n_body - off_cur)
                tokens[i:i + n_cur] = body_tokens[off_cur:off_cur + n_cur]
                off += n_cur
                n_rest -= n_cur
                i += n_cur
                body_tok_num = n_cur

            if off == n_doc and n_rest > 0 and not doc_end_written:
                tokens[i] = self.doc_end_tok
                i += 1
                n_rest -= 1
                doc_end_written = True

            chunk = self.TokChunk(
                docid=docid, tokens=tokens, offset=offset, docid_tok_num=len(docid_tokens),
                offset_tok_num=len(off_tokens), title_tok_num=title_tok_num, body_tok_num=body_tok_num,
            )
            chunks.append(chunk)
        return chunks

    def _split_approx(self, docid: int, docid_tokens: list[int], title_tokens: list[int], body_tokens: list[int]) -> list[TokChunk]:
        n_title, n_body = len(title_tokens), len(body_tokens)
        n_doc = n_title + n_body
        n_emb_tokens = self.n_emb_tokens - (1 + 3 + 3)
        n_embs = n_doc // n_emb_tokens
        n_mod = n_doc % n_emb_tokens
        if n_embs == 0 or n_mod >= n_emb_tokens // 2:
            n_embs += 1
        embs_offsets = np.linspace(0, n_doc, n_embs + 1, dtype=int)
        head_tokens = [self.doc_beg_tok, *docid_tokens]
        chunks = []
        for i in range(n_embs):
            i1, i2 = int(embs_offsets[i]), int(embs_offsets[i + 1])
            offset = i1
            off_tokens = self.tokenizer(str(offset))['input_ids']
            off_tokens = [self.offset_beg_tok, *off_tokens, self.offset_end_tok]
            tokens = head_tokens + off_tokens
            title_tok_num = 0
            if i1 < n_title:
                i1_cur = i1
                i2_cur = min(i2, n_title)
                tokens.extend(title_tokens[i1_cur:i2_cur])
                title_tok_num = i2_cur - i1_cur
            body_tok_num = 0
            if i2 > n_title:
                i1_cur = max(i1 - n_title, 0)
                i2_cur = min(i2 - n_title, n_body)
                tokens.extend(body_tokens[i1_cur:i2_cur])
                body_tok_num = i2_cur - i1_cur
            if i2 == n_doc:
                tokens.append(self.doc_end_tok)
            tokens = np.array(tokens, dtype=np.int32)
            chunk = self.TokChunk(
                docid=docid, tokens=tokens, offset=offset, docid_tok_num=len(docid_tokens),
                offset_tok_num=len(off_tokens), title_tok_num=title_tok_num, body_tok_num=body_tok_num,
            )
            chunks.append(chunk)
        return chunks

    def process_doc(self, docid: int, doc: dict[str, str]):
        title, body = doc['title'], doc['text']
        title_tokens, body_tokens = self.tokenizer(title), self.tokenizer(body)
        title_tokens, body_tokens = title_tokens['input_ids'], body_tokens['input_ids']
        docid_tokens = self.tokenizer(str(docid))['input_ids']
        docid_tokens = [self.id_beg_tok, *docid_tokens, self.id_end_tok]
        title_tokens = [self.title_beg_tok, *title_tokens, self.title_end_tok]
        body_tokens = [self.body_beg_tok, *body_tokens, self.body_end_tok]
        if self.fixed_size:
            chunks = self._split_fixed(docid, docid_tokens, title_tokens, body_tokens)
        else:
            chunks = self._split_approx(docid, docid_tokens, title_tokens, body_tokens)

        self.chunks.extend(chunks)
        self.docs_processed_num += 1
        self._write_data_if_needed()
