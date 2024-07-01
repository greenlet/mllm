from pathlib import Path
from typing import Optional, Union, Iterable

import numpy as np
import pandas as pd

from transformers import PreTrainedTokenizer


FIXED_SUFFIX = 'fixed'
NONFIXED_SUFFIX = 'nonfixed'
Strings = Union[str, Iterable[str]]


class CustomToken:
    name: str
    repr: str
    special: bool
    ind: int = 0

    def __init__(self, name: str, special: bool):
        self.name = name
        self.repr = f'<|{name}|>'
        self.special = special

    def set_ind(self, ind: int):
        self.ind = ind

    def __str__(self) -> str:
        return f'{self.__class__.__name__}. Name: {self.name}. Repr: {self.repr}. Ind: {self.ind}'

    def __repr__(self) -> str:
        return str(self)


TokDict = dict[str, CustomToken]


def add_pref_post(s: str, pref: str = '', post: str = '') -> str:
    if pref:
        s = pref if not s else f'{pref}_{s}'
    if post:
        s = post if not s else f'{s}_{post}'
    return s


def gen_tokens(tokens_names: list[str], prefix: Strings = '', postfix: Strings = '', special: bool = False) -> TokDict:
    prefixes = [prefix] if type(prefix) == str else prefix
    postfixes = [postfix] if type(postfix) == str else postfix
    res = {}
    for tname in tokens_names:
        for prefix in prefixes:
            for postfix in postfixes:
                tnamepp = add_pref_post(tname, prefix, postfix)
                token = CustomToken(tnamepp, special)
                res[token.name] = token
    return res


def add_tokens(tokenizer: PreTrainedTokenizer, tokens: TokDict):
    for t in tokens.values():
        if t.special:
            tokenizer.add_special_tokens({f'{t.name}_token': t.repr})
        else:
            tokenizer.add_tokens(t.repr)
        t.set_ind((len(tokenizer) - 1))


def gen_doc_tokens() -> TokDict:
    tokens = gen_tokens(['', 'id', 'offset', 'title', 'body'], prefix='doc', postfix=['begin', 'end'])
    return tokens


def gen_special_tokens() -> TokDict:
    tokens = gen_tokens(['pad'], special=True)
    return tokens


def gen_dec_tokens() -> TokDict:
    tokens = gen_tokens(['query'], postfix=['begin', 'end'])
    return tokens


def gen_all_tokens(tokenizer: Optional[PreTrainedTokenizer] = None) -> TokDict:
    tokens = {
        **gen_doc_tokens(),
        **gen_special_tokens(),
        **gen_dec_tokens(),
    }
    if tokenizer is not None:
        add_tokens(tokenizer, tokens)
    return tokens


def gen_out_subdir(emb_chunk_size: int, fixed_size: bool) -> str:
    fixed_suffix = FIXED_SUFFIX if fixed_size else NONFIXED_SUFFIX
    return f'ch_{emb_chunk_size:03d}_{fixed_suffix}'


def parse_out_subdir(subdir: str) -> tuple[int, bool]:
    parts = subdir.split('_')
    emb_chunk_size, fixed_suffix = int(parts[1]), parts[2]
    if fixed_suffix == FIXED_SUFFIX:
        fixed_size = True
    elif fixed_suffix == NONFIXED_SUFFIX:
        fixed_size = False
    else:
        raise ValueError(f'Unexpected fixed_suffix="{fixed_suffix}" in subdir name "{subdir}". '
                         f'Expected values: {FIXED_SUFFIX}, {NONFIXED_SUFFIX}')
    return emb_chunk_size, fixed_size


def gen_ds_fnames(doc_id_min: int, doc_id_max: int) -> tuple[str, str, str]:
    fname_base = f'docs_{doc_id_min:07d}-{doc_id_max:07d}'
    return f'{fname_base}.csv', f'{fname_base}_tokens.np', f'{fname_base}_chunk_sizes.np'


def split_doc_embs(n_doc: int, n_emb_tokens: int) -> np.ndarray:
    n_embs = n_doc // n_emb_tokens
    n_mod = n_doc % n_emb_tokens
    if n_embs == 0 or n_mod >= n_emb_tokens // 2:
        n_embs += 1
    embs_offsets = np.linspace(0, n_doc, n_embs + 1, dtype=int)
    return embs_offsets


def calc_max_inp_size(n_emb_tokens: int) -> int:
    return n_emb_tokens + n_emb_tokens // 2 - 1


class ChunkTokenizer:
    class TokChunk:
        docid: int
        tokens: np.ndarray
        offset: int
        docid_tok_num: int
        offset_tok_num: int
        title_tok_num: int
        body_tok_num: int
        title_beg_ind: int
        title_end_ind: int
        body_beg_ind: int
        body_end_ind: int

        def __init__(self, docid: int, tokens: np.ndarray, offset: int, docid_tok_num: int,
                     offset_tok_num: int, title_tok_num: int, body_tok_num: int,
                     title_beg_ind: int = -1, title_end_ind: int = -1, body_beg_ind: int = -1, body_end_ind: int = -1):
            self.docid = docid
            self.tokens = tokens
            self.offset = offset
            self.docid_tok_num = docid_tok_num
            self.offset_tok_num = offset_tok_num
            self.title_tok_num = title_tok_num
            self.body_tok_num = body_tok_num
            assert title_beg_ind == title_end_ind == -1 or 0 < title_beg_ind < title_end_ind, f'title_beg_ind = {title_beg_ind}, title_end_ind = {title_end_ind}'
            assert body_beg_ind == body_end_ind == -1 or 0 < body_beg_ind < body_end_ind, f'body_beg_ind = {body_beg_ind}, body_end_ind = {body_end_ind}'
            self.title_beg_ind = title_beg_ind
            self.title_end_ind = title_end_ind
            self.body_beg_ind = body_beg_ind
            self.body_end_ind = body_end_ind

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

    def write_data(self):
        if not self.chunks:
            return
        keys = 'docid', 'offset', 'tok_num', 'docid_tok_num', \
            'offset_tok_num', 'title_tok_num', 'body_tok_num', \
            'title_beg_ind', 'title_end_ind', 'body_beg_ind', 'body_end_ind', \
            'doc_id_min', 'doc_id_max', 'doc_id_off'

        assert self.dir_out is not None
        self.dir_out.mkdir(parents=True, exist_ok=True)
        n_chunks = len(self.chunks)
        data = {k: [0] * n_chunks for k in keys}
        if self.fixed_size:
            tokens = np.empty(shape=(n_chunks, self.n_emb_tokens), dtype=np.int32)
            chunk_sizes, tok_off = None, None
        else:
            n_tok_all = sum(len(ch.tokens) for ch in self.chunks)
            tokens = np.empty(shape=(n_tok_all,), dtype=np.int32)
            chunk_sizes = np.empty(shape=(n_chunks,), dtype=np.int32)
            tok_off = 0

        doc_id_min, doc_id_max = self.chunks[0].docid, self.chunks[-1].docid
        for i, chunk in enumerate(self.chunks):
            data['docid'][i] = chunk.docid
            data['offset'][i] = chunk.offset
            data['tok_num'][i] = len(chunk.tokens)
            data['docid_tok_num'][i] = chunk.docid_tok_num
            data['offset_tok_num'][i] = chunk.offset_tok_num
            data['title_tok_num'][i] = chunk.title_tok_num
            data['body_tok_num'][i] = chunk.body_tok_num
            data['title_beg_ind'][i] = chunk.title_beg_ind
            data['title_end_ind'][i] = chunk.title_end_ind
            data['body_beg_ind'][i] = chunk.body_beg_ind
            data['body_end_ind'][i] = chunk.body_end_ind
            data['doc_id_min'][i] = doc_id_min
            data['doc_id_max'][i] = doc_id_max
            data['doc_id_off'][i] = i

            if self.fixed_size:
                tokens[i] = chunk.tokens
            else:
                chunk_size = len(chunk.tokens)
                tokens[tok_off:tok_off + chunk_size] = chunk.tokens
                tok_off += chunk_size
                chunk_sizes[i] = chunk_size

        df_fname, tokens_fname, sizes_fname = gen_ds_fnames(doc_id_min, doc_id_max)

        tokens_fpath = self.dir_out / tokens_fname
        tokens.tofile(tokens_fpath)
        if not self.fixed_size:
            chunk_sizes_fpath = self.dir_out / sizes_fname
            chunk_sizes.tofile(chunk_sizes_fpath)

        df = pd.DataFrame(data)
        df_fpath = self.dir_out / df_fname
        df.to_csv(df_fpath, index=False)

        self.chunks = []
        self.docs_processed_num = 0

    def _write_data_if_needed(self):
        if self.docs_write_num <= 0 or self.dir_out is None or self.docs_processed_num < self.docs_write_num:
            return
        self.write_data()

    def _split_fixed(self, docid: int, docid_tokens: list[int], title_tokens: list[int], body_tokens: list[int]) -> list[TokChunk]:
        n_docid = len(docid_tokens)
        n_title, n_body = len(title_tokens), len(body_tokens)
        n_doc = n_title + n_body
        off = 0
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
            title_beg_ind = title_end_ind = -1
            if off < n_title:
                off_cur = off
                n_cur = min(n_rest, n_title - off_cur)
                tokens[i:i + n_cur] = title_tokens[off_cur:off_cur + n_cur]
                j1 = i + (title_tokens[off_cur] == self.title_beg_tok)
                j2 = i + n_cur - (title_tokens[off_cur + n_cur - 1] == self.title_end_tok)
                if j1 < j2:
                    title_beg_ind, title_end_ind = j1, j2
                off += n_cur
                n_rest -= n_cur
                i += n_cur
                title_tok_num = n_cur

            body_tok_num = 0
            body_beg_ind = body_end_ind = -1
            if off >= n_title and n_rest > 0:
                off_cur = off - n_title
                n_cur = min(n_rest, n_body - off_cur)
                tokens[i:i + n_cur] = body_tokens[off_cur:off_cur + n_cur]
                j1 = i + (body_tokens[off_cur] == self.body_beg_tok)
                j2 = i + n_cur - (body_tokens[off_cur + n_cur - 1] == self.body_end_tok)
                if j1 < j2:
                    body_beg_ind, body_end_ind = j1, j2
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
                title_beg_ind=title_beg_ind, title_end_ind=title_end_ind, body_beg_ind=body_beg_ind, body_end_ind=body_end_ind,
            )
            chunks.append(chunk)
        return chunks

    def _split_approx(self, docid: int, docid_tokens: list[int], title_tokens: list[int], body_tokens: list[int]) -> list[TokChunk]:
        n_title, n_body = len(title_tokens), len(body_tokens)
        n_doc = n_title + n_body
        n_emb_tokens = self.n_emb_tokens - (1 + 3 + 3)
        embs_offsets = split_doc_embs(n_doc, n_emb_tokens)
        head_tokens = [self.doc_beg_tok, *docid_tokens]
        chunks = []
        for i in range(embs_offsets.shape[0]):
            i1, i2 = int(embs_offsets[i]), int(embs_offsets[i + 1])
            offset = i1
            off_tokens = self.tokenizer(str(offset))['input_ids']
            off_tokens = [self.offset_beg_tok, *off_tokens, self.offset_end_tok]
            tokens = head_tokens + off_tokens
            title_tok_num = 0
            title_beg_ind = title_end_ind = -1
            if i1 < n_title:
                i1_cur = i1
                i2_cur = min(i2, n_title)
                tokens.extend(title_tokens[i1_cur:i2_cur])
                title_tok_num = i2_cur - i1_cur
                j1 = len(tokens) - title_tok_num + (title_tokens[i1_cur] == self.title_beg_tok)
                j2 = len(tokens) - (title_tokens[i2_cur - 1] == self.title_end_tok)
                if j1 < j2:
                    title_beg_ind, title_end_ind = j1, j2

            body_tok_num = 0
            body_beg_ind = body_end_ind = -1
            if i2 > n_title:
                i1_cur = max(i1 - n_title, 0)
                i2_cur = min(i2 - n_title, n_body)
                tokens.extend(body_tokens[i1_cur:i2_cur])
                body_tok_num = i2_cur - i1_cur
                j1 = len(tokens) - body_tok_num + (body_tokens[i1_cur] == self.body_beg_tok)
                j2 = len(tokens) - (body_tokens[i2_cur - 1] == self.body_end_tok)
                if j1 < j2:
                    body_beg_ind, body_end_ind = j1, j2

            if i2 == n_doc:
                tokens.append(self.doc_end_tok)

            tokens = np.array(tokens, dtype=np.int32)
            chunk = self.TokChunk(
                docid=docid, tokens=tokens, offset=offset, docid_tok_num=len(docid_tokens),
                offset_tok_num=len(off_tokens), title_tok_num=title_tok_num, body_tok_num=body_tok_num,
                title_beg_ind=title_beg_ind, title_end_ind=title_end_ind, body_beg_ind=body_beg_ind, body_end_ind=body_end_ind,
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

