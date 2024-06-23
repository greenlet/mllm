
import unittest as ut
from dataclasses import dataclass
from typing import Union

import numpy as np

from transformers import PreTrainedTokenizer, GPT2Tokenizer

from mllm.tokenization.chunk_tokenizer import ChunkTokenizer, CustomToken, gen_add_doc_tokens

TXT_01 = 'x'
TXT_02 = 'Bear beer boaring boring'
TXT_03 = 'A planet is a large, rounded body that is generally required to be in orbit around a star, stellar remnant or brown dwarf.'
TXT_04 = '''The word planet probably comes from the Greek πλανήται (planḗtai), meaning 'wanderers'. In antiquity, this word referred to the Sun, Moon, and five points of light visible to the naked eye that moved across the background of the stars—namely, Mercury, Venus, Mars, Jupiter, and Saturn. Planets have historically had religious associations: multiple cultures identified celestial bodies with gods, and these connections with mythology and folklore persist in the schemes for naming newly discovered Solar System bodies. Earth itself was recognized as a planet when heliocentrism supplanted geocentrism during the 16th and 17th centuries.'''


@dataclass(kw_only=True)
class TChunkCase:
    docid_tok_num: int
    offset: int
    offset_tok_num: int
    title_tok_num: int
    body_tok_num: int
    doc_tokens: list[int]


@dataclass(kw_only=True)
class TCase:
    docid: int
    title: str
    body: str
    docid_tokens: list[int]
    title_tokens: list[int]
    body_tokens: list[int]
    n_emb_tokens: int
    fixed_size: bool
    chunks: list[TChunkCase]


class TestChunkTokenizer(ut.TestCase):
    tokenizer: GPT2Tokenizer
    tokens_dict: dict[str, CustomToken]

    @property
    def doc_beg_tok(self) -> int:
        return self.tokens_dict['doc_begin'].ind

    @property
    def doc_end_tok(self) -> int:
        return self.tokens_dict['doc_end'].ind

    @property
    def id_beg_tok(self) -> int:
        return self.tokens_dict['doc_id_begin'].ind

    @property
    def id_end_tok(self) -> int:
        return self.tokens_dict['doc_id_end'].ind

    @property
    def offset_beg_tok(self) -> int:
        return self.tokens_dict['doc_offset_begin'].ind

    @property
    def offset_end_tok(self) -> int:
        return self.tokens_dict['doc_offset_end'].ind

    @property
    def title_beg_tok(self) -> int:
        return self.tokens_dict['doc_title_begin'].ind

    @property
    def title_end_tok(self) -> int:
        return self.tokens_dict['doc_title_end'].ind

    @property
    def body_beg_tok(self) -> int:
        return self.tokens_dict['doc_body_begin'].ind

    @property
    def body_end_tok(self) -> int:
        return self.tokens_dict['doc_body_end'].ind

    @property
    def pad_tok(self) -> int:
        return self.tokens_dict['pad'].ind

    def _tok(self, txt: str) -> list[int]:
        return self.tokenizer(txt)['input_ids']

    def _tok_docid(self, docid: Union[str, int]) -> list[int]:
        return [self.id_beg_tok, *self._tok(str(docid)), self.id_end_tok]

    def _tok_offset(self, offset: Union[str, int]) -> list[int]:
        return [self.offset_beg_tok, *self._tok(str(offset)), self.offset_end_tok]

    def _tok_title(self, title: str) -> list[int]:
        return [self.title_beg_tok, *self._tok(title), self.title_end_tok]

    def _tok_body(self, body: str) -> list[int]:
        return [self.body_beg_tok, *self._tok(body), self.body_end_tok]

    def _doc(self, title: str, body: str) -> dict[str, str]:
        return {'title': title, 'text': body}

    def _str_case_chunk(self, case: TCase, cchunk: 'TChunkCase', chunk: ChunkTokenizer.TokChunk) -> str:
        res = [
            f'Doc id: {case.docid}',
            f'Title: {case.title}',
            f'Body: {case.body}',
            f'n_emb_tokens: {case.n_emb_tokens}. fixed_size: {case.fixed_size}',
            f'-- Expected - Actual',
            f'-- docid_tok_num: {cchunk.docid_tok_num} - {chunk.docid_tok_num}',
            f'-- offset_tok_num: {cchunk.offset_tok_num} - {chunk.offset_tok_num}',
            f'-- title_tok_num: {cchunk.title_tok_num} - {chunk.title_tok_num}',
            f'-- body_tok_num: {cchunk.body_tok_num} - {chunk.body_tok_num}',
            f'Expected tokens: {cchunk.doc_tokens}',
            f'Actual tokens:   {chunk.tokens}',
        ]
        return '\n  '.join(res)

    def setUp(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokens_dict = gen_add_doc_tokens(self.tokenizer)

    def test_01(self):
        for n_emb_tokens in 20, 50, 100:
            for fixed_size in True, False:
                ch_tkz = ChunkTokenizer(self.tokens_dict, self.tokenizer, n_emb_tokens, fixed_size)

                docid, title, body = 0, 'title', TXT_01
                docid_tokens = self._tok_docid(docid)
                title_tokens = self._tok_title(title)
                body_tokens = self._tok_body(body)
                case = TCase(
                    docid=docid, title=title, body=body, docid_tokens=docid_tokens, title_tokens=title_tokens,
                    body_tokens=body_tokens, n_emb_tokens=n_emb_tokens, fixed_size=fixed_size, chunks=[],
                )
                offset = 0
                offset_tokens = self._tok_offset(offset)
                case.chunks.append(TChunkCase(
                    docid_tok_num=len(docid_tokens), offset=offset, offset_tok_num=len(offset_tokens),
                    title_tok_num=len(title_tokens), body_tok_num=len(body_tokens), doc_tokens=[
                        self.doc_beg_tok,
                        *docid_tokens,
                        *offset_tokens,
                        *title_tokens,
                        *body_tokens,
                        self.doc_end_tok,
                    ]
                ))
                ch_tkz.process_doc(docid, self._doc(title, body))
                self.assertEqual(1, len(ch_tkz.chunks))
                self.assertEqual(3, len(docid_tokens))
                self.assertEqual(3, len(offset_tokens))
                self.assertEqual(3, len(title_tokens))
                self.assertEqual(3, len(body_tokens))

                chunk, cchunk = ch_tkz.chunks[0], case.chunks[0]
                self.assertEqual(cchunk.docid_tok_num, chunk.docid_tok_num)
                self.assertEqual(cchunk.offset_tok_num, chunk.offset_tok_num)
                self.assertEqual(cchunk.body_tok_num, chunk.body_tok_num)
                self.assertEqual(cchunk.title_tok_num, chunk.title_tok_num)

                n_doc_tok = len(cchunk.doc_tokens)
                self.assertEqual(chunk.docid_tok_num, chunk.docid_tok_num, self._str_case_chunk(case, cchunk, chunk))
                self.assertEqual(chunk.offset_tok_num, chunk.offset_tok_num, self._str_case_chunk(case, cchunk, chunk))
                self.assertEqual(chunk.title_tok_num, chunk.title_tok_num, self._str_case_chunk(case, cchunk, chunk))
                self.assertEqual(chunk.body_tok_num, chunk.body_tok_num, self._str_case_chunk(case, cchunk, chunk))
                self.assertEqual(True, np.alltrue(cchunk.doc_tokens == chunk.tokens[:n_doc_tok]), self._str_case_chunk(case, cchunk, chunk))
                self.assertEqual(True, np.alltrue(self.pad_tok == chunk.tokens[n_doc_tok:]), self._str_case_chunk(case, cchunk, chunk))

    def test_02(self):
        docid, title, body = 0, 'title 1', TXT_02
        docid_tokens = self._tok_docid(docid)
        title_tokens =  self._tok_title(title)
        body_tokens = self._tok_body(body)
        prefix_tokens = [self.doc_beg_tok, *docid_tokens]
        docid_tok_num = len(docid_tokens)

        cases = []
        # ------- Test case 1 ------- #
        case = TCase(
            docid=docid, title=title, body=body, docid_tokens=docid_tokens, title_tokens=title_tokens,
            body_tokens=body_tokens, n_emb_tokens=9, fixed_size=True, chunks=[],
        )
        offset = 0
        offset_tokens = self._tok_offset(offset)
        case.chunks.append(TChunkCase(
            docid_tok_num=docid_tok_num, offset=offset, offset_tok_num=len(offset_tokens),
            title_tok_num=2, body_tok_num=0, doc_tokens=[
                *prefix_tokens,
                *offset_tokens,
                *title_tokens[:2],
            ]
        ))
        offset = 2
        offset_tokens = self._tok_offset(offset)
        case.chunks.append(TChunkCase(
            docid_tok_num=docid_tok_num, offset=offset, offset_tok_num=len(offset_tokens),
            title_tok_num=2, body_tok_num=0, doc_tokens=[
                *prefix_tokens,
                *offset_tokens,
                *title_tokens[2:],
            ]
        ))
        offset = 4
        offset_tokens = self._tok_offset(offset)
        case.chunks.append(TChunkCase(
            docid_tok_num=docid_tok_num, offset=offset, offset_tok_num=len(offset_tokens),
            title_tok_num=0, body_tok_num=2, doc_tokens=[
                *prefix_tokens,
                *offset_tokens,
                *body_tokens[:2],
            ]
        ))
        offset = 6
        offset_tokens = self._tok_offset(offset)
        case.chunks.append(TChunkCase(
            docid_tok_num=docid_tok_num, offset=offset, offset_tok_num=len(offset_tokens),
            title_tok_num=0, body_tok_num=2, doc_tokens=[
                *prefix_tokens,
                *offset_tokens,
                *body_tokens[2:4],
            ]
        ))
        offset = 8
        offset_tokens = self._tok_offset(offset)
        case.chunks.append(TChunkCase(
            docid_tok_num=docid_tok_num, offset=offset, offset_tok_num=len(offset_tokens),
            title_tok_num=0, body_tok_num=2, doc_tokens=[
                *prefix_tokens,
                *offset_tokens,
                *body_tokens[4:6],
            ]
        ))
        offset = 10
        offset_tokens = self._tok_offset(offset)
        case.chunks.append(TChunkCase(
            docid_tok_num=docid_tok_num, offset=offset, offset_tok_num=len(offset_tokens),
            title_tok_num=0, body_tok_num=1, doc_tokens=[
                *prefix_tokens,
                *offset_tokens,
                *body_tokens[6:],
                self.doc_end_tok,
            ]
        ))
        cases.append(case)

        # ------- Test case 2 ------- #
        case = TCase(
            docid=docid, title=title, body=body, docid_tokens=docid_tokens, title_tokens=title_tokens,
            body_tokens=body_tokens, n_emb_tokens=11, fixed_size=True, chunks=[],
        )
        offset = 0
        offset_tokens = self._tok_offset(offset)
        case.chunks.append(TChunkCase(
            docid_tok_num=docid_tok_num, offset=offset, offset_tok_num=len(offset_tokens),
            title_tok_num=4, body_tok_num=0, doc_tokens=[
                *prefix_tokens,
                *offset_tokens,
                *title_tokens,
            ]
        ))
        offset = 4
        offset_tokens = self._tok_offset(offset)
        case.chunks.append(TChunkCase(
            docid_tok_num=docid_tok_num, offset=offset, offset_tok_num=len(offset_tokens),
            title_tok_num=0, body_tok_num=4, doc_tokens=[
                *prefix_tokens,
                *offset_tokens,
                *body_tokens[:4],
            ]
        ))
        offset = 8
        offset_tokens = self._tok_offset(offset)
        case.chunks.append(TChunkCase(
            docid_tok_num=docid_tok_num, offset=offset, offset_tok_num=len(offset_tokens),
            title_tok_num=0, body_tok_num=3, doc_tokens=[
                *prefix_tokens,
                *offset_tokens,
                *body_tokens[4:],
                self.doc_end_tok,
            ]
        ))
        cases.append(case)

        # ------- Test case 3 ------- #
        case = TCase(
            docid=docid, title=title, body=body, docid_tokens=docid_tokens, title_tokens=title_tokens,
            body_tokens=body_tokens, n_emb_tokens=12, fixed_size=True, chunks=[],
        )
        offset = 0
        offset_tokens = self._tok_offset(offset)
        case.chunks.append(TChunkCase(
            docid_tok_num=docid_tok_num, offset=offset, offset_tok_num=len(offset_tokens),
            title_tok_num=4, body_tok_num=1, doc_tokens=[
                *prefix_tokens,
                *offset_tokens,
                *title_tokens,
                *body_tokens[:1],
            ]
        ))
        offset = 5
        offset_tokens = self._tok_offset(offset)
        case.chunks.append(TChunkCase(
            docid_tok_num=docid_tok_num, offset=offset, offset_tok_num=len(offset_tokens),
            title_tok_num=0, body_tok_num=5, doc_tokens=[
                *prefix_tokens,
                *offset_tokens,
                *body_tokens[1:6],
            ]
        ))
        offset = 10
        offset_tokens = self._tok_offset(offset)
        case.chunks.append(TChunkCase(
            docid_tok_num=docid_tok_num, offset=offset, offset_tok_num=len(offset_tokens),
            title_tok_num=0, body_tok_num=1, doc_tokens=[
                *prefix_tokens,
                *offset_tokens,
                *body_tokens[6:],
                self.doc_end_tok,
            ]
        ))
        cases.append(case)

        # ------- Test case 4 ------- #
        case = TCase(
            docid=docid, title=title, body=body, docid_tokens=docid_tokens, title_tokens=title_tokens,
            body_tokens=body_tokens, n_emb_tokens=10, fixed_size=False, chunks=[],
        )
        offset = 0
        offset_tokens = self._tok_offset(offset)
        case.chunks.append(TChunkCase(
            docid_tok_num=docid_tok_num, offset=offset, offset_tok_num=len(offset_tokens),
            title_tok_num=2, body_tok_num=0, doc_tokens=[
                *prefix_tokens,
                *offset_tokens,
                *title_tokens[:2],
            ]
        ))
        offset = 2
        offset_tokens = self._tok_offset(offset)
        case.chunks.append(TChunkCase(
            docid_tok_num=docid_tok_num, offset=offset, offset_tok_num=len(offset_tokens),
            title_tok_num=2, body_tok_num=1, doc_tokens=[
                *prefix_tokens,
                *offset_tokens,
                *title_tokens[2:],
                *body_tokens[:1],
            ]
        ))
        offset = 5
        offset_tokens = self._tok_offset(offset)
        case.chunks.append(TChunkCase(
            docid_tok_num=docid_tok_num, offset=offset, offset_tok_num=len(offset_tokens),
            title_tok_num=0, body_tok_num=3, doc_tokens=[
                *prefix_tokens,
                *offset_tokens,
                *body_tokens[1:4],
            ]
        ))
        offset = 8
        offset_tokens = self._tok_offset(offset)
        case.chunks.append(TChunkCase(
            docid_tok_num=docid_tok_num, offset=offset, offset_tok_num=len(offset_tokens),
            title_tok_num=0, body_tok_num=3, doc_tokens=[
                *prefix_tokens,
                *offset_tokens,
                *body_tokens[4:],
                self.doc_end_tok,
            ]
        ))
        cases.append(case)

        for case in cases:
            ch_tkz = ChunkTokenizer(self.tokens_dict, self.tokenizer, case.n_emb_tokens, case.fixed_size)
            ch_tkz.process_doc(0, self._doc(title, body))
            self.assertEqual(1 + 2, docid_tok_num)
            self.assertEqual(2 + 2, len(title_tokens))
            self.assertEqual(5 + 2, len(body_tokens))
            self.assertEqual(len(case.chunks), len(ch_tkz.chunks))

            for i, chunk in enumerate(ch_tkz.chunks):
                cchunk = case.chunks[i]
                self.assertEqual(docid_tok_num, chunk.docid_tok_num)
                self.assertEqual(cchunk.offset_tok_num, chunk.offset_tok_num, self._str_case_chunk(case, cchunk, chunk))
                self.assertEqual(cchunk.body_tok_num, chunk.body_tok_num, self._str_case_chunk(case, cchunk, chunk))
                self.assertEqual(cchunk.title_tok_num, chunk.title_tok_num, self._str_case_chunk(case, cchunk, chunk))

                self.assertEqual(True, np.alltrue(chunk.tokens[:len(cchunk.doc_tokens)] == cchunk.doc_tokens), self._str_case_chunk(case, cchunk, chunk))
                self.assertEqual(True, np.alltrue(chunk.tokens[len(cchunk.doc_tokens):] == self.pad_tok), self._str_case_chunk(case, cchunk, chunk))


if __name__ == '__main__':
    ut.main()

