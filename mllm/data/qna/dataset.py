"""Common QnA batch and base dataset for MixedDecoder training.

Each QnA item has:
  - context: passage text chunked into segments for BERT encoder
  - question: formatted as prompt for the decoder
  - answer: target text for autoregressive generation (may be empty for unanswerable questions)

Multi-turn datasets additionally carry conversation history.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Generator, List, Optional, Tuple

import numpy as np
import torch
from transformers import PreTrainedTokenizer


class QnaDatasetType(str, Enum):
    """Registry of all supported QnA datasets."""
    SQUAD_V2 = 'squad_v2'
    NATURAL_QUESTIONS = 'natural_questions'
    TRIVIAQA = 'triviaqa'
    NEWSQA = 'newsqa'
    MRQA = 'mrqa'
    ADVERSARIALQA = 'adversarialqa'
    SQUAD_V1 = 'squad_v1'
    QUAC = 'quac'
    COQA = 'coqa'


@dataclass(kw_only=True)
class QnaBatch:
    """Batch of QnA items prepared for MixedDecoder training.

    Context chunks are encoded by the BERT encoder to produce CLS embeddings.
    These embeddings serve as the context prefix for the autoregressive decoder.
    The prompt encodes the question (and optionally conversation history for multi-turn).
    The answer is the generation target (empty string for unanswerable questions).
    """
    # --- Context ---
    # (total_chunks_in_batch, inp_len) — all context chunk tokens across batch items
    ctx_chunks_toks: torch.Tensor
    # (total_chunks_in_batch, inp_len) — attention masks for context chunks
    ctx_chunks_att_mask: torch.Tensor
    # Number of context chunks per QnA item (list of length batch_size)
    ctx_chunk_counts: List[int]

    # --- Prompt (question / conversation history) ---
    # (batch_size, max_prompt_len) — tokenized prompt, right-padded
    prompt_toks: torch.Tensor
    # (batch_size, max_prompt_len) — attention mask for prompts
    prompt_att_mask: torch.Tensor
    # Actual token lengths of each prompt before padding (list of length batch_size)
    prompt_lengths: List[int]

    # --- Answer ---
    # (batch_size, max_ans_len) — target answer tokens (with special tokens)
    ans_toks: torch.Tensor
    # (batch_size, max_ans_len) — attention mask for answer tokens
    ans_att_mask: torch.Tensor

    # --- Metadata ---
    # Whether each item is answerable (list of length batch_size)
    answerable: List[bool]


class QnaBaseDataset:
    """Base class for QnA datasets providing common tokenization and batching logic.

    Subclasses must implement:
      - _load_data(): load and prepare the dataset-specific dataframe / data structure
      - _get_item(idx): return (context, prompt_text, answer_text, is_answerable) for a single item
    """

    NO_ANSWER_TEXT = 'noanswer'

    def __init__(
            self, tkz: PreTrainedTokenizer,
            inp_len: int, max_chunks: int, max_ans_toks: int = 100,
            max_prompt_toks: int = 100, device: Optional[torch.device] = None,
    ):
        self.tkz = tkz
        self.inp_len = inp_len
        self.max_chunks = max_chunks
        self.max_ans_toks = max_ans_toks
        self.max_prompt_toks = max_prompt_toks
        self.device = device if device is not None else torch.device('cpu')
        self.pad_token_id = tkz.pad_token_id
        self.cls_token_id = tkz.cls_token_id
        self.sep_token_id = tkz.sep_token_id

        # Single-turn prompt formatting: 'Question: {q} Answer:'
        self.prompt_prefix_toks = self.tkz('Question: ', add_special_tokens=False).input_ids
        self.prompt_suffix_toks = self.tkz(' Answer:', add_special_tokens=False).input_ids
        self.prompt_budget = (
            self.max_prompt_toks
            - len(self.prompt_prefix_toks)
            - len(self.prompt_suffix_toks)
        )
        assert self.prompt_budget > 0, (
            f'Not enough max_prompt_toks to fit prefix and suffix. '
            f'prompt_budget={self.prompt_budget}. '
            f'prefix_len={len(self.prompt_prefix_toks)}. '
            f'suffix_len={len(self.prompt_suffix_toks)}. '
            f'max_prompt_toks={self.max_prompt_toks}.'
        )

        # Multi-turn prompt formatting: 'Q: {q0} A: {a0}. Q: {q1} A: {a1}. ... Q: {qn} A:'
        self.mt_q_prefix_toks = self.tkz('Q: ', add_special_tokens=False).input_ids
        self.mt_a_infix_toks = self.tkz(' A: ', add_special_tokens=False).input_ids
        self.mt_turn_sep_toks = self.tkz('. ', add_special_tokens=False).input_ids
        self.mt_suffix_toks = self.tkz(' A:', add_special_tokens=False).input_ids

        # To be set by subclasses
        self.inds: np.ndarray = np.array([], dtype=np.int64)

    @property
    def size(self) -> int:
        return len(self.inds)

    def __len__(self) -> int:
        return self.size

    # --- Methods to override in subclasses ---

    def _get_item(self, idx: int) -> Tuple[str, List[str], List[str], bool]:
        """Return (context, questions, answers, is_answerable) for item at index idx.

        questions and answers are parallel lists of equal length.
        For single-turn datasets: questions = [q], answers = [a].
        For multi-turn datasets: questions = [q_0, ..., q_n], answers = [a_0, ..., a_n]
        where answers[-1] is the generation target.

        For unanswerable items, answers[-1] should be NO_ANSWER_TEXT.
        """
        raise NotImplementedError

    # --- Common tokenization ---

    def chunk_context(self, context: str) -> List[List[int]]:
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

    def tokenize_prompt(self, questions: List[str], answers: List[str]) -> List[int]:
        """Tokenize prompt from question/answer lists.

        Single-turn (len == 1): 'Question: {q} Answer:'
        Multi-turn  (len >  1): 'Q: {q0} A: {a0}. Q: {q1} A: {a1}. ... Q: {qn} A:'

        questions and answers have equal length.  answers[-1] is the generation
        target and does NOT appear in the prompt.

        For multi-turn, history turns are added one by one from oldest to newest.
        If a turn would exceed the token budget, remaining history is skipped.
        The current question (questions[-1]) is always included.
        """
        assert len(questions) == len(answers)
        n = len(questions)

        if n == 1:
            # Single-turn: existing format
            q_toks = self.tkz(questions[0], add_special_tokens=False).input_ids
            if len(q_toks) > self.prompt_budget:
                q_toks = q_toks[-self.prompt_budget:]
            return self.prompt_prefix_toks + q_toks + self.prompt_suffix_toks

        # Multi-turn ----------------------------------------------------------
        # Current question block (always included)
        cur_q_toks = self.tkz(questions[-1], add_special_tokens=False).input_ids
        cur_block = self.mt_q_prefix_toks + cur_q_toks + self.mt_suffix_toks

        # Truncate current question from the left if it alone exceeds budget
        if len(cur_block) > self.max_prompt_toks:
            budget_for_q = (
                self.max_prompt_toks
                - len(self.mt_q_prefix_toks)
                - len(self.mt_suffix_toks)
            )
            cur_q_toks = cur_q_toks[-max(budget_for_q, 1):]
            cur_block = self.mt_q_prefix_toks + cur_q_toks + self.mt_suffix_toks

        budget = self.max_prompt_toks - len(cur_block)

        # Add history turns from oldest to newest
        history_toks: List[int] = []
        for j in range(n - 2, -1, -1):
            q_toks = self.tkz(questions[j], add_special_tokens=False).input_ids
            a_toks = self.tkz(answers[j], add_special_tokens=False).input_ids
            turn_toks = (
                self.mt_q_prefix_toks + q_toks
                + self.mt_a_infix_toks + a_toks
                + self.mt_turn_sep_toks
            )
            if len(turn_toks) > budget:
                break
            history_toks.extend(turn_toks)
            budget -= len(turn_toks)

        return history_toks + cur_block

    def tokenize_answer(self, answer: str) -> List[int]:
        """Tokenize answer with SEP token at the end."""
        toks = self.tkz(answer, add_special_tokens=True).input_ids[1:]
        if len(toks) > self.max_ans_toks:
            toks = toks[:self.max_ans_toks]
        return toks

    # --- Batching ---

    def get_batch(self, inds: List[int]) -> QnaBatch:
        batch_size = len(inds)

        all_chunks: List[List[int]] = []
        chunk_counts: List[int] = []
        prompt_toks_list: List[List[int]] = []
        ans_toks_list: List[List[int]] = []
        answerable_list: List[bool] = []

        for idx in inds:
            context, questions, answers, is_answerable = self._get_item(idx)

            chunks = self.chunk_context(context)
            all_chunks.extend(chunks)
            chunk_counts.append(len(chunks))

            prompt_toks_list.append(self.tokenize_prompt(questions, answers))
            ans_toks_list.append(self.tokenize_answer(answers[-1]))
            answerable_list.append(is_answerable)

        # Pad context chunks to inp_len
        total_chunks = len(all_chunks)
        ctx_chunks_t = torch.full(
            (total_chunks, self.inp_len), self.pad_token_id, dtype=torch.long, device=self.device,
        )
        ctx_chunks_att = torch.zeros(
            (total_chunks, self.inp_len), dtype=torch.long, device=self.device,
        )
        for i, chunk in enumerate(all_chunks):
            n = min(len(chunk), self.inp_len)
            ctx_chunks_t[i, :n] = torch.tensor(chunk[:n], dtype=torch.long, device=self.device)
            ctx_chunks_att[i, :n] = 1

        # Right-pad prompts
        prompt_lengths = [len(p) for p in prompt_toks_list]
        max_prompt_len = max(prompt_lengths)
        prompt_t = torch.full(
            (batch_size, max_prompt_len), self.pad_token_id, dtype=torch.long, device=self.device,
        )
        prompt_att = torch.zeros(
            (batch_size, max_prompt_len), dtype=torch.long, device=self.device,
        )
        for i, toks in enumerate(prompt_toks_list):
            n = len(toks)
            prompt_t[i, :n] = torch.tensor(toks, dtype=torch.long, device=self.device)
            prompt_att[i, :n] = 1

        # Pad answers
        max_ans_len = max(len(a) for a in ans_toks_list)
        ans_t = torch.full(
            (batch_size, max_ans_len), self.pad_token_id, dtype=torch.long, device=self.device,
        )
        ans_att = torch.zeros(
            (batch_size, max_ans_len), dtype=torch.long, device=self.device,
        )
        for i, toks in enumerate(ans_toks_list):
            n = len(toks)
            ans_t[i, :n] = torch.tensor(toks, dtype=torch.long, device=self.device)
            ans_att[i, :n] = 1

        return QnaBatch(
            ctx_chunks_toks=ctx_chunks_t,
            ctx_chunks_att_mask=ctx_chunks_att,
            ctx_chunk_counts=chunk_counts,
            prompt_toks=prompt_t,
            prompt_att_mask=prompt_att,
            prompt_lengths=prompt_lengths,
            ans_toks=ans_t,
            ans_att_mask=ans_att,
            answerable=answerable_list,
        )

    def shuffle(self, seed: Optional[int] = None) -> 'QnaBaseDataset':
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.inds)
        else:
            np.random.shuffle(self.inds)
        return self
