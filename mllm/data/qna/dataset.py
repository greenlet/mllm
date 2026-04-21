"""Common QnA batch and base dataset for MixedDecoder training.

Each QnA item has:
  - context: passage text chunked into segments for BERT encoder
  - question: formatted as prompt for the decoder
  - answer: target text for autoregressive generation (may be empty for unanswerable questions)

Multi-turn datasets additionally carry conversation history.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from mllm.data.qna.batch import QnaBatch


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


QNA_DATASETS_DEFAULT = (
    QnaDatasetType.SQUAD_V2,
    QnaDatasetType.NATURAL_QUESTIONS,
    QnaDatasetType.TRIVIAQA,
    QnaDatasetType.NEWSQA,
    QnaDatasetType.MRQA,
    QnaDatasetType.ADVERSARIALQA,
    QnaDatasetType.QUAC,
    QnaDatasetType.COQA,
)


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


class QnaDatasetAgg:
    """Aggregator that combines multiple QnaBaseDataset instances under a single global index.

    Maintains a mapping from global index to (dataset_index, local_index) so that
    ``_get_item``, ``get_batch``, and ``shuffle`` work transparently across all
    underlying datasets.
    """

    def __init__(self, datasets: List[QnaBaseDataset], device: Optional[torch.device] = None):
        assert len(datasets) > 0, 'QnaDatasetAgg requires at least one dataset'
        self.datasets = datasets
        self.device = device if device is not None else torch.device('cpu')

        # Build 2-column numpy array: [ds_idx, local_idx] per global index
        ds_lens = [len(ds) for ds in self.datasets]
        total = sum(ds_lens)
        self._map = np.empty((total, 2), dtype=np.int64)
        offset = 0
        for ds_idx, ds_len in enumerate(ds_lens):
            self._map[offset:offset + ds_len, 0] = ds_idx
            self._map[offset:offset + ds_len, 1] = np.arange(ds_len, dtype=np.int64)
            offset += ds_len

        self.inds = np.arange(total, dtype=np.int64)

    @property
    def size(self) -> int:
        return len(self.inds)

    def __len__(self) -> int:
        return self.size

    def _get_item(self, idx: int) -> Tuple[str, List[str], List[str], bool]:
        ds_idx, local_idx = self._map[idx]
        return self.datasets[ds_idx]._get_item(local_idx)

    def get_batch(self, inds: List[int]) -> QnaBatch:
        """Build a batch by delegating to the first dataset's batching logic."""
        # Group items by dataset to resolve global → local indices, but
        # use the first dataset's tokenization/batching (all share the same config).
        ds0 = self.datasets[0]

        batch_size = len(inds)
        all_chunks: List[List[int]] = []
        chunk_counts: List[int] = []
        prompt_toks_list: List[List[int]] = []
        ans_toks_list: List[List[int]] = []
        answerable_list: List[bool] = []

        for idx in inds:
            context, questions, answers, is_answerable = self._get_item(idx)

            chunks = ds0.chunk_context(context)
            all_chunks.extend(chunks)
            chunk_counts.append(len(chunks))

            prompt_toks_list.append(ds0.tokenize_prompt(questions, answers))
            ans_toks_list.append(ds0.tokenize_answer(answers[-1]))
            answerable_list.append(is_answerable)

        # Pad context chunks to inp_len
        total_chunks = len(all_chunks)
        ctx_chunks_t = torch.full(
            (total_chunks, ds0.inp_len), ds0.pad_token_id, dtype=torch.long, device=self.device,
        )
        ctx_chunks_att = torch.zeros(
            (total_chunks, ds0.inp_len), dtype=torch.long, device=self.device,
        )
        for i, chunk in enumerate(all_chunks):
            n = min(len(chunk), ds0.inp_len)
            ctx_chunks_t[i, :n] = torch.tensor(chunk[:n], dtype=torch.long, device=self.device)
            ctx_chunks_att[i, :n] = 1

        # Right-pad prompts
        prompt_lengths = [len(p) for p in prompt_toks_list]
        max_prompt_len = max(prompt_lengths)
        prompt_t = torch.full(
            (batch_size, max_prompt_len), ds0.pad_token_id, dtype=torch.long, device=self.device,
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
            (batch_size, max_ans_len), ds0.pad_token_id, dtype=torch.long, device=self.device,
        )
        ans_att = torch.zeros(
            (batch_size, max_ans_len), dtype=torch.long, device=self.device,
        )
        for i, toks in enumerate(ans_toks_list):
            n = len(toks)
            ans_t[i, :n] = torch.tensor(toks, dtype=torch.long, device=self.device)
            ans_att[i, :n] = 1

        ds_src_list = [int(self._map[idx, 0]) for idx in inds]

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
            ds_src=ds_src_list,
        )

    @property
    def ds_names(self) -> List[str]:
        """Return human-readable dataset names, one per underlying dataset."""
        return [type(ds).__name__ for ds in self.datasets]

    def ds_name(self, ds_idx: int) -> str:
        """Return the class name for a dataset by its index."""
        return type(self.datasets[ds_idx]).__name__

    def shuffle(self, seed: Optional[int] = None) -> 'QnaDatasetAgg':
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.inds)
        else:
            np.random.shuffle(self.inds)
        return self


def load_qna_datasets(
        tkz: PreTrainedTokenizer,
        inp_len: int,
        max_chunks: int,
        max_ans_toks: int = 100,
        max_prompt_toks: int = 100,
        device: Optional[torch.device] = None,
        cache_dir: str | Path | None = None,
        sources: Tuple[QnaDatasetType, ...] = QNA_DATASETS_DEFAULT,
) -> Tuple[QnaDatasetAgg, QnaDatasetAgg]:
    """Load QnA datasets and return aggregated train/val splits.

    Args:
        sources: Tuple of QnaDatasetType values to load. Defaults to all except SQuAD v1.

    Returns:
        (train_agg, val_agg): QnaDatasetAgg for train and validation splits.
    """
    from mllm.data.qna.ds_01_squad_v2 import SquadV2Dataset, load_squad_v2
    from mllm.data.qna.ds_02_natural_questions import NaturalQuestionsDataset, load_nq
    from mllm.data.qna.ds_03_triviaqa import TriviaQADataset, load_triviaqa
    from mllm.data.qna.ds_04_newsqa import NewsqaDataset, load_newsqa
    from mllm.data.qna.ds_05_quac import QuacDataset, load_quac
    from mllm.data.qna.ds_06_coqa import CoqaDataset, load_coqa
    from mllm.data.qna.ds_07_mrqa import MrqaDataset, load_mrqa
    from mllm.data.qna.ds_08_adversarialqa import AdversarialqaDataset, load_adversarialqa
    from mllm.data.qna.ds_09_squad_v1 import SquadV1Dataset, load_squad_v1

    _REGISTRY = {
        QnaDatasetType.SQUAD_V2: (load_squad_v2, SquadV2Dataset),
        QnaDatasetType.NATURAL_QUESTIONS: (load_nq, NaturalQuestionsDataset),
        QnaDatasetType.TRIVIAQA: (load_triviaqa, TriviaQADataset),
        QnaDatasetType.NEWSQA: (load_newsqa, NewsqaDataset),
        QnaDatasetType.QUAC: (load_quac, QuacDataset),
        QnaDatasetType.COQA: (load_coqa, CoqaDataset),
        QnaDatasetType.MRQA: (load_mrqa, MrqaDataset),
        QnaDatasetType.ADVERSARIALQA: (load_adversarialqa, AdversarialqaDataset),
        QnaDatasetType.SQUAD_V1: (load_squad_v1, SquadV1Dataset),
    }

    ds_kwargs = dict(tkz=tkz, inp_len=inp_len, max_chunks=max_chunks,
                     max_ans_toks=max_ans_toks, max_prompt_toks=max_prompt_toks, device=device)

    train_datasets: List[QnaBaseDataset] = []
    val_datasets: List[QnaBaseDataset] = []

    for src in sources:
        load_fn, ds_cls = _REGISTRY[src]
        ds_train_hf, ds_val_hf = load_fn(cache_dir=cache_dir)
        train_datasets.append(ds_cls(ds=ds_train_hf, **ds_kwargs))
        val_datasets.append(ds_cls(ds=ds_val_hf, **ds_kwargs))

    return QnaDatasetAgg(train_datasets, device=device), QnaDatasetAgg(val_datasets, device=device)


def create_qna_dataloader(
        ds_agg: QnaDatasetAgg,
        batch_size: int,
        shuffle: bool = True,
        inds: Optional[np.ndarray] = None,
) -> Generator[QnaBatch, None, None]:
    """Create infinite-loop generator yielding QnaBatch instances.

    Args:
        ds_agg: Aggregated QnA dataset.
        batch_size: Number of items per batch.
        shuffle: Whether to shuffle indices after each full pass.
        inds: Optional index array to use instead of ds_agg.inds.
    """
    from torch import distributed as dist

    cur_inds = inds if inds is not None else ds_agg.inds
    cur_inds = cur_inds.copy()
    n = len(cur_inds)
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f'R{rank}. Create QnaCiteDataset dataloader. total={n}. batch_size={batch_size}. shuffle={shuffle}.')
    start_ind = 0
    while True:
        end_ind = min(start_ind + batch_size, n)
        batch_inds = cur_inds[start_ind:end_ind].tolist()
        if len(batch_inds) < batch_size:
            batch_inds += cur_inds[:(batch_size - len(batch_inds))].tolist()
        batch = ds_agg.get_batch(batch_inds)
        if end_ind == n and shuffle:
                np.random.shuffle(cur_inds)
        yield batch
        start_ind = end_ind % n

