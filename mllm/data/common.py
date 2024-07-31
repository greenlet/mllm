import itertools
import itertools as it
from typing import Optional

import numpy as np
import torch

from mllm.tokenization.chunk_tokenizer import split_doc_embs


class DocsBatch:
    docs_chunks: dict[int, list[np.ndarray]]
    target_doc_id: int
    target_tokens: list[list[int]]
    pad_tok: int
    qbeg_tok: int
    qend_tok: int
    emb_chunk_size: int
    device: torch.device
    docs_chunks_padded: np.ndarray
    target_chunks_padded: np.ndarray
    target_mask: np.ndarray
    fixed_size: bool
    docs_chunks_padded_tf: Optional[torch.Tensor] = None
    target_chunks_padded_tf: Optional[torch.Tensor] = None
    target_mask_tf: Optional[torch.Tensor] = None
    device: Optional[torch.device] = None

    def __init__(self, docs_chunks: dict[int, list[np.ndarray]], target_doc_id: int, target_tokens: list[list[int]],
                 pad_tok: int, qbeg_tok: int, qend_tok: int, emb_chunk_size: int, fixed_size: bool,
                 device: Optional[torch.device] = None):
        self.docs_chunks = docs_chunks
        self.target_doc_id = target_doc_id
        self.target_tokens = target_tokens
        self.pad_tok = pad_tok
        self.qbeg_tok = qbeg_tok
        self.qend_tok = qend_tok
        self.emb_chunk_size = emb_chunk_size
        self.fixed_size = fixed_size
        self.device = device
        self.calc_np()

    @staticmethod
    def _sync_chunks_mask(chunks: list[list[int]], mask: np.ndarray):
        n_chunks = len(chunks)
        i0 = None
        for i in range(n_chunks):
            if mask[i] and i0 is None:
                i0 = i
            if i0 is not None and i - i0 < n_chunks:
                assert mask[i], f'mask = {mask}. n_chunks = {n_chunks}.'
                mask[i] = len(chunks[i - i0]) > 0

    def calc_np(self):
        docs_chunks = []
        target_chunk_off, target_chunk_sz = 0, 0
        for doc_id, chunks in self.docs_chunks.items():
            if target_chunk_sz == 0:
                if doc_id == self.target_doc_id:
                    target_chunk_sz = len(chunks)
                else:
                    target_chunk_off += len(chunks)
            docs_chunks.extend(chunks)

        target_mask = np.full(len(docs_chunks), False, dtype=bool)
        target_mask[target_chunk_off:target_chunk_off + target_chunk_sz] = True

        # self._sync_chunks_mask(self.target_tokens, target_mask)

        target_tokens = list(itertools.chain(*self.target_tokens))
        target_tokens = [self.qbeg_tok, *target_tokens, self.qend_tok]

        target_embs_offsets = split_doc_embs(len(target_tokens), self.emb_chunk_size, self.fixed_size)
        n_target_chunks = len(target_embs_offsets) - 1
        target_chunks = []
        for i in range(n_target_chunks):
            chunk = target_tokens[target_embs_offsets[i]:target_embs_offsets[i + 1]]
            target_chunks.append(chunk)

        n_batch_chunks = len(docs_chunks)
        max_chank_sz = max(len(chunk) for chunk in it.chain(docs_chunks, target_chunks))

        docs_chunks_padded = np.full((n_batch_chunks, max_chank_sz), self.pad_tok, dtype=np.int32)
        for i_chunk, chunk in enumerate(docs_chunks):
            docs_chunks_padded[i_chunk, :len(chunk)] = chunk

        target_chunks_padded = np.full((n_target_chunks, max_chank_sz), self.pad_tok, dtype=np.int32)
        for i_chunk, chunk in enumerate(target_chunks):
            target_chunks_padded[i_chunk, :len(chunk)] = chunk

        self.docs_chunks_padded = docs_chunks_padded
        self.target_chunks_padded = target_chunks_padded
        self.target_mask = target_mask

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        res = torch.from_numpy(arr)
        if self.device is not None:
            res = res.to(self.device)
        return res

    def gen_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.docs_chunks_padded_tf is None:
            self.docs_chunks_padded_tf, self.target_chunks_padded_tf, self.target_mask_tf = \
                map(self._to_tensor, (self.docs_chunks_padded, self.target_chunks_padded, self.target_mask))
        return self.docs_chunks_padded_tf, self.target_chunks_padded_tf, self.target_mask_tf
