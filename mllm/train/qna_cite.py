"""QnA batch and dataset iterator for MixedDecoder training with SQuAD v2 data.

For each QnA item the context is chunked into segments of `inp_len` tokens (up to `max_chunks` chunks).
All chunks are encoded by the encoder to produce CLS embeddings.
The question is formatted as "Question: {q} Answer:" and used as the prompt.
The answer text is the target for autoregressive generation.
"""

from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from transformers import PreTrainedTokenizer

from mllm.data.itsquadv2 import get_squadv2_df
from mllm.data.utils import split_df


@dataclass(kw_only=True)
class QnaCiteBatch:
    """Batch of QnA items prepared for MixedDecoder training.

    Context chunks are encoded by the BERT encoder to produce CLS embeddings.
    These embeddings serve as the context prefix for the autoregressive decoder.
    """
    # (total_chunks_in_batch, inp_len) - all context chunk tokens concatenated across batch items
    ctx_chunks_toks: torch.Tensor
    # (total_chunks_in_batch, inp_len) - attention masks for context chunks
    ctx_chunks_att_mask: torch.Tensor
    # Number of context chunks per QnA item (list of length batch_size)
    ctx_chunk_counts: List[int]
    # (batch_size, max_prompt_len) - tokenized "Question: {q} Answer:" right-padded
    prompt_toks: torch.Tensor
    # (batch_size, max_prompt_len) - attention mask for prompts
    prompt_att_mask: torch.Tensor
    # Actual token lengths of each prompt before padding (list of length batch_size)
    prompt_lengths: List[int]
    # (batch_size, ans_len) - target answer tokens (with special tokens)
    ans_toks: torch.Tensor
    # (batch_size, ans_len) - attention mask for answer tokens
    ans_att_mask: torch.Tensor


class QnaCiteDataset:
    """Dataset that yields QnaCiteBatch from SQuAD v2 data.

    Each context is split into chunks of `inp_len` tokens.
    Each chunk gets CLS prepended so that BERT encoding produces a meaningful [CLS] embedding.
    """

    def __init__(
            self, df: pd.DataFrame, inds: np.ndarray, tkz: PreTrainedTokenizer,
            inp_len: int, max_chunks: int, max_ans_toks: int = 100,
            max_prompt_toks: int = 100, device: Optional[torch.device] = None,
    ):
        self.df = df
        self.inds = inds.copy()
        self.tkz = tkz
        self.inp_len = inp_len
        self.max_chunks = max_chunks
        self.max_ans_toks = max_ans_toks
        self.max_prompt_toks = max_prompt_toks
        self.device = device if device is not None else torch.device('cpu')
        self.pad_token_id = tkz.pad_token_id
        self.cls_token_id = tkz.cls_token_id
        self.sep_token_id = tkz.sep_token_id
        self.size = len(self.inds)
        self.prompt_prefix_toks = self.tkz('Question: ', add_special_tokens=False).input_ids
        self.prompt_suffix_toks = self.tkz(' Answer:', add_special_tokens=False).input_ids
        self.prompt_budget = self.max_prompt_toks - len(self.prompt_prefix_toks) - len(self.prompt_suffix_toks)
        assert self.prompt_budget > 0, f'Not enough max_prompt_toks to fit prefix and suffix. prompt_budget={self.prompt_budget}. prefix_len={len(self.prompt_prefix_toks)}. suffix_len={len(self.prompt_suffix_toks)}. max_prompt_toks={self.max_prompt_toks}.'


    def __len__(self):
        return self.size

    def _chunk_context(self, context: str) -> List[List[int]]:
        """Tokenize context and split into chunks, each starting with CLS and ending with SEP."""
        ctx_toks = self.tkz(context, add_special_tokens=False).input_ids
        chunk_content_len = self.inp_len - 2  # reserve 1 for CLS and 1 for SEP
        chunks = []
        for start in range(0, len(ctx_toks), chunk_content_len):
            content = ctx_toks[start:start + chunk_content_len]
            chunk = [self.cls_token_id] + content + [self.sep_token_id]
            chunks.append(chunk)
            if len(chunks) >= self.max_chunks:
                break
        return chunks

    def _tokenize_prompt(self, question: str) -> List[int]:
        """Tokenize question into prompt format: 'Question: {q} Answer:'.

        If the full prompt does not fit into max_prompt_toks, the question is
        truncated from the left so that the *last* part of the question is kept,
        preserving the 'Question: ' prefix and ' Answer:' suffix.
        """
        q_toks = self.tkz(question, add_special_tokens=False).input_ids

        if len(q_toks) > self.prompt_budget:
            q_toks = q_toks[-self.prompt_budget:]  # keep the last (most relevant) part
        return self.prompt_prefix_toks + q_toks + self.prompt_suffix_toks

    def _tokenize_answer(self, answer: str) -> List[int]:
        """Tokenize answer with SEP token at the end."""
        toks = self.tkz(answer, add_special_tokens=True).input_ids[1:]
        if len(toks) > self.max_ans_toks:
            toks = toks[:self.max_ans_toks]
        return toks

    def get_batch(self, inds: List[int]) -> QnaCiteBatch:
        batch_size = len(inds)

        all_chunks: List[List[int]] = []
        chunk_counts: List[int] = []
        prompt_toks_list: List[List[int]] = []
        ans_toks_list: List[List[int]] = []

        for idx in inds:
            row = self.df.iloc[idx]
            context = row['context']
            question = row['question']
            answers = row['answers']['text']
            n_answers = len(answers)
            assert n_answers > 0, f'Expected at least one answer for SQuAD v2 item, but got n_answers={n_answers} for idx={idx}.'
            # randomly sample one answer if multiple are present
            answer = np.random.choice(answers)

            chunks = self._chunk_context(context)
            all_chunks.extend(chunks)
            chunk_counts.append(len(chunks))

            prompt_toks_list.append(self._tokenize_prompt(question))
            ans_toks_list.append(self._tokenize_answer(answer))

        # Pad context chunks to inp_len
        total_chunks = len(all_chunks)
        ctx_chunks_t = torch.full((total_chunks, self.inp_len), self.pad_token_id, dtype=torch.long, device=self.device)
        ctx_chunks_att = torch.zeros((total_chunks, self.inp_len), dtype=torch.long, device=self.device)
        for i, chunk in enumerate(all_chunks):
            n = min(len(chunk), self.inp_len)
            ctx_chunks_t[i, :n] = torch.tensor(chunk[:n], dtype=torch.long, device=self.device)
            ctx_chunks_att[i, :n] = 1

        # Right-pad prompts; store actual lengths for per-sample sequence building
        prompt_lengths = [len(p) for p in prompt_toks_list]
        max_prompt_len = max(prompt_lengths)
        prompt_t = torch.full((batch_size, max_prompt_len), self.pad_token_id, dtype=torch.long, device=self.device)
        prompt_att = torch.zeros((batch_size, max_prompt_len), dtype=torch.long, device=self.device)
        for i, toks in enumerate(prompt_toks_list):
            n = len(toks)
            prompt_t[i, :n] = torch.tensor(toks, dtype=torch.long, device=self.device)
            prompt_att[i, :n] = 1

        # Pad answers
        max_ans_len = max(len(a) for a in ans_toks_list)
        ans_t = torch.full((batch_size, max_ans_len), self.pad_token_id, dtype=torch.long, device=self.device)
        ans_att = torch.zeros((batch_size, max_ans_len), dtype=torch.long, device=self.device)
        for i, toks in enumerate(ans_toks_list):
            n = len(toks)
            ans_t[i, :n] = torch.tensor(toks, dtype=torch.long, device=self.device)
            ans_att[i, :n] = 1

        return QnaCiteBatch(
            ctx_chunks_toks=ctx_chunks_t,
            ctx_chunks_att_mask=ctx_chunks_att,
            ctx_chunk_counts=chunk_counts,
            prompt_toks=prompt_t,
            prompt_att_mask=prompt_att,
            prompt_lengths=prompt_lengths,
            ans_toks=ans_t,
            ans_att_mask=ans_att,
        )

    def shuffle(self, seed: Optional[int] = None) -> 'QnaCiteDataset':
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.inds)
        else:
            np.random.shuffle(self.inds)
        return self


def load_split_squadv2(
        exclude_empty_answers: bool = True, val_ratio: float = 0.05, random_seed: int = 333,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load SQuAD v2 and split into train/val indices."""
    df = get_squadv2_df(exclude_empty_answers=exclude_empty_answers)
    n_total = len(df)
    # Re-index to use iloc correctly
    df = df.reset_index(drop=True)
    inds = np.arange(n_total)
    # if random_seed is not None:
    #     rng = np.random.default_rng(random_seed)
    #     rng.shuffle(inds)
    # else:
    #     np.random.shuffle(inds)
    n_val = int(n_total * val_ratio)
    inds_train = inds[n_val:].copy()
    inds_val = inds[:n_val].copy()
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        rng.shuffle(inds_train)
        rng.shuffle(inds_val)
    else:
        np.random.shuffle(inds_train)
        np.random.shuffle(inds_val)
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f'R{rank}. SQuAD v2 n_total={n_total}. n_train={len(inds_train)}. n_val={len(inds_val)}.')
    return df, inds_train, inds_val


def create_qna_cite_dataloader(
        dataset: QnaCiteDataset, batch_size: int, shuffle: bool = True,
) -> Generator[QnaCiteBatch, None, None]:
    """Create infinite-loop generator yielding QnaCiteBatch instances."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f'R{rank}. Create QnaCiteDataset dataloader. batch_size={batch_size}. shuffle={shuffle}.')
    start_ind = 0
    while True:
        end_ind = min(start_ind + batch_size, len(dataset))
        inds = dataset.inds[start_ind:end_ind].tolist()
        if len(inds) < batch_size:
            inds += dataset.inds[:(batch_size - len(inds))].tolist()
        batch = dataset.get_batch(inds)
        if end_ind == len(dataset):
            print(f'R{rank}. Shuffle QnaCite dataset')
            dataset.shuffle()
        yield batch
        start_ind = end_ind % len(dataset)
