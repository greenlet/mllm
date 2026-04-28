"""Next-token prediction batch and dataset iterator for MixedDecoder training with Wikipedia data.

For each Wikipedia article:
1. Tokenize the full article text.
2. Determine how many complete chunks fit while reserving at least ``min_next_toks`` tokens
   for the prediction target.
3. Randomly select a contiguous window of chunks as context (encoded by BERT → CLS embeddings).
4. The tokens immediately following the context window become the autoregressive target.
A fixed "Continue:" prompt bridges context embeddings and target tokens in the decoder input.
"""

from dataclasses import dataclass, field
from math import floor
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizer

from mllm.data.wiki.itwiki import get_split_wiki_ds


@dataclass(kw_only=True)
class NextTokBatch:
    """Batch of next-token prediction items prepared for MixedDecoder training.

    Context chunks are encoded by the BERT encoder to produce CLS embeddings.
    These embeddings serve as the context prefix for the autoregressive decoder.
    """
    # (total_chunks_in_batch, inp_len) – all context chunk tokens concatenated across batch items
    ctx_chunks_toks: torch.Tensor
    # (total_chunks_in_batch, inp_len) – attention masks for context chunks
    ctx_chunks_att_mask: torch.Tensor
    # Number of context chunks per item (list of length batch_size)
    ctx_chunk_counts: List[int]
    # (batch_size, prompt_len) – tokenized "Continue:" prompt, right-padded
    prompt_toks: torch.Tensor
    # (batch_size, prompt_len) – attention mask for prompts
    prompt_att_mask: torch.Tensor
    # Actual token lengths of each prompt before padding (list of length batch_size)
    prompt_lengths: List[int]
    # (batch_size, max_target_len) – target tokens (with CLS at start, SEP at end, right-padded)
    target_toks: torch.Tensor
    # (batch_size, max_target_len) – attention mask for target tokens
    target_att_mask: torch.Tensor


class NextTokWikiDataset:
    """Dataset that yields :class:`NextTokBatch` from Wikipedia articles.

    Each article is tokenized into sub-word tokens.  A contiguous window of
    ``win_size`` chunks (each ``inp_len`` tokens with CLS/SEP) is selected as
    context, and the subsequent tokens form the prediction target.
    """

    def __init__(
            self, ds: Dataset, inds: np.ndarray, tkz_enc: PreTrainedTokenizer,
            inp_len: int, min_next_toks: int,
            emb_win_min_size: int, emb_win_max_size: int,
            max_target_toks: int = 128, device: Optional[torch.device] = None,
            tkz_dec: Optional[PreTrainedTokenizer] = None,
    ):
        self.ds = ds
        self.inds = inds.copy()
        self.tkz_enc = tkz_enc
        self.tkz_dec = tkz_dec if tkz_dec is not None else tkz_enc
        self.tkz = tkz_enc  # backward-compat alias
        self.inp_len = inp_len
        self.min_next_toks = min_next_toks
        self.emb_win_min_size = max(emb_win_min_size, 1)
        self.emb_win_max_size = max(emb_win_max_size, 1)
        self.max_target_toks = max_target_toks
        self.device = device if device is not None else torch.device('cpu')
        self.enc_pad_token_id = tkz_enc.pad_token_id
        self.dec_pad_token_id = self.tkz_dec.pad_token_id
        if self.dec_pad_token_id is None:
            self.dec_pad_token_id = self.tkz_dec.eos_token_id
        self.cls_token_id = tkz_enc.cls_token_id
        self.sep_token_id = tkz_enc.sep_token_id
        self.chunk_content_len = inp_len - 2  # reserve 1 for CLS and 1 for SEP
        self.size = len(self.inds)
        # Decoder-side specials for wrapping the target sequence.
        if self.tkz_dec.cls_token_id is not None and self.tkz_dec.sep_token_id is not None:
            # BERT-like decoder: wrap with CLS .. SEP.
            self.dec_target_prefix: List[int] = [self.tkz_dec.cls_token_id]
            self.dec_target_suffix: List[int] = [self.tkz_dec.sep_token_id]
        elif self.tkz_dec.eos_token_id is not None:
            # GPT-2: only end-of-text token; use as target ending.
            self.dec_target_prefix = []
            self.dec_target_suffix = [self.tkz_dec.eos_token_id]
        else:
            self.dec_target_prefix = []
            self.dec_target_suffix = []
        # Reserve room for prefix+suffix in the target budget.
        self.dec_target_content_budget = max(
            self.max_target_toks - len(self.dec_target_prefix) - len(self.dec_target_suffix), 1,
        )
        # Fixed prompt tokenized once with the decoder tokenizer.
        self.prompt_toks_fixed: List[int] = self.tkz_dec('Continue:', add_special_tokens=False).input_ids
        # Internal pointer for skipping short docs
        self._ptr = 0

    def __len__(self) -> int:
        return self.size

    def _tokenize_doc(self, text: str) -> List[int]:
        return self.tkz_enc(text, add_special_tokens=False).input_ids

    def _make_chunks(self, doc_toks: List[int], start: int, win_size: int) -> List[List[int]]:
        """Split ``win_size`` consecutive content-length segments into CLS+content+SEP chunks."""
        chunks: List[List[int]] = []
        for i in range(win_size):
            seg_start = start + i * self.chunk_content_len
            seg_end = seg_start + self.chunk_content_len
            content = doc_toks[seg_start:seg_end]
            chunk = [self.cls_token_id] + content + [self.sep_token_id]
            chunks.append(chunk)
        return chunks

    def _sample_item(self) -> Optional[Tuple[List[List[int]], List[int]]]:
        """Try to sample one valid item from the current pointer position.

        Returns ``None`` if the current document is too short (caller should
        advance the pointer and retry).
        """
        idx = self.inds[self._ptr].item()
        row = self.ds[idx]
        text: str = row['text']
        doc_toks = self._tokenize_doc(text)
        doc_toks_num = len(doc_toks)

        doc_chunks_num = max(floor((doc_toks_num - self.min_next_toks) / self.chunk_content_len), 0)
        if doc_chunks_num < self.emb_win_min_size:
            return None

        win_max_size_new = min(doc_chunks_num, self.emb_win_max_size)
        win_size = np.random.randint(self.emb_win_min_size, win_max_size_new + 1)

        ctx_tok_count = win_size * self.chunk_content_len
        max_start = doc_toks_num - ctx_tok_count - self.min_next_toks
        start = np.random.randint(0, max_start + 1)

        chunks = self._make_chunks(doc_toks, start, win_size)

        # Target: tokens right after the context window. We slice the encoder
        # tokenization, decode to text, and re-tokenize with the decoder vocab.
        tgt_start = start + ctx_tok_count
        tgt_end = min(tgt_start + self.max_target_toks * 4, doc_toks_num)
        tgt_text = self.tkz_enc.decode(doc_toks[tgt_start:tgt_end], skip_special_tokens=True)
        tgt_content = self.tkz_dec(tgt_text, add_special_tokens=False).input_ids
        tgt_content = tgt_content[:self.dec_target_content_budget]
        target = self.dec_target_prefix + tgt_content + self.dec_target_suffix

        return chunks, target

    def _advance_ptr(self) -> None:
        self._ptr += 1
        if self._ptr >= self.size:
            np.random.shuffle(self.inds)
            self._ptr = 0

    def _sample_valid_item(self) -> Tuple[List[List[int]], List[int]]:
        """Keep sampling until a document long enough is found."""
        while True:
            result = self._sample_item()
            self._advance_ptr()
            if result is not None:
                return result

    def get_batch(self, batch_size: int) -> NextTokBatch:
        all_chunks: List[List[int]] = []
        chunk_counts: List[int] = []
        prompt_toks_list: List[List[int]] = []
        target_toks_list: List[List[int]] = []

        for _ in range(batch_size):
            chunks, target = self._sample_valid_item()
            all_chunks.extend(chunks)
            chunk_counts.append(len(chunks))
            prompt_toks_list.append(list(self.prompt_toks_fixed))
            target_toks_list.append(target)

        # Pad context chunks to inp_len (encoder vocab)
        total_chunks = len(all_chunks)
        ctx_chunks_t = torch.full((total_chunks, self.inp_len), self.enc_pad_token_id, dtype=torch.long, device=self.device)
        ctx_chunks_att = torch.zeros((total_chunks, self.inp_len), dtype=torch.long, device=self.device)
        for i, chunk in enumerate(all_chunks):
            n = min(len(chunk), self.inp_len)
            ctx_chunks_t[i, :n] = torch.tensor(chunk[:n], dtype=torch.long, device=self.device)
            ctx_chunks_att[i, :n] = 1

        # Right-pad prompts (decoder vocab; fixed length, but keep interface consistent)
        prompt_lengths = [len(p) for p in prompt_toks_list]
        max_prompt_len = max(prompt_lengths)
        prompt_t = torch.full((batch_size, max_prompt_len), self.dec_pad_token_id, dtype=torch.long, device=self.device)
        prompt_att = torch.zeros((batch_size, max_prompt_len), dtype=torch.long, device=self.device)
        for i, toks in enumerate(prompt_toks_list):
            n = len(toks)
            prompt_t[i, :n] = torch.tensor(toks, dtype=torch.long, device=self.device)
            prompt_att[i, :n] = 1

        # Pad target tokens (decoder vocab)
        max_target_len = max(len(t) for t in target_toks_list)
        target_t = torch.full((batch_size, max_target_len), self.dec_pad_token_id, dtype=torch.long, device=self.device)
        target_att = torch.zeros((batch_size, max_target_len), dtype=torch.long, device=self.device)
        for i, toks in enumerate(target_toks_list):
            n = len(toks)
            target_t[i, :n] = torch.tensor(toks, dtype=torch.long, device=self.device)
            target_att[i, :n] = 1

        return NextTokBatch(
            ctx_chunks_toks=ctx_chunks_t,
            ctx_chunks_att_mask=ctx_chunks_att,
            ctx_chunk_counts=chunk_counts,
            prompt_toks=prompt_t,
            prompt_att_mask=prompt_att,
            prompt_lengths=prompt_lengths,
            target_toks=target_t,
            target_att_mask=target_att,
        )

    def shuffle(self, seed: Optional[int] = None) -> 'NextTokWikiDataset':
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.inds)
        else:
            np.random.shuffle(self.inds)
        self._ptr = 0
        return self


def load_split_wiki_for_next(
        data_path: Path, val_ratio: float = 0.05, random_seed: int = 100,
) -> Tuple[Dataset, np.ndarray, np.ndarray]:
    """Load Wikipedia and split into train/val index arrays.

    Returns:
        Tuple of (ds, inds_train, inds_val) where *ds* is the HuggingFace
        ``Dataset`` object and the index arrays select documents for each split.
    """
    ds, inds_train, inds_val = get_split_wiki_ds(data_path, val_ratio=val_ratio, rand_seed=random_seed)
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f'R{rank}. Wikipedia for NextTok: n_total={len(ds)}. n_train={len(inds_train)}. n_val={len(inds_val)}.')
    return ds, inds_train, inds_val


def create_next_tok_dataloader(
        dataset: NextTokWikiDataset, batch_size: int,
) -> Generator[NextTokBatch, None, None]:
    """Create infinite-loop generator yielding :class:`NextTokBatch` instances."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f'R{rank}. Create NextTokWikiDataset dataloader. batch_size={batch_size}.')
    while True:
        batch = dataset.get_batch(batch_size)
        yield batch
