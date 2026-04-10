"""Natural Questions dataset loader for QnA MixedDecoder training.

HuggingFace: google-research-datasets/natural_questions
Splits: train (~307K), validation (~7.8K)
Columns: id, question (dict with 'text'), document (tokenized HTML), annotations (short_answers, long_answer, yes_no_answer)
Context: long_answer span as focused context (fallback to full doc); HTML tokens randomly included or stripped
Answer: randomly chosen between short_answer and long_answer (when both exist); HTML always stripped from answers
Unanswerable: yes — no short/long answer and no yes_no → answer = 'noanswer'
Multi-turn: no
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HfDataset
from transformers import PreTrainedTokenizer

from mllm.data.qna.dataset import QnaBaseDataset


NQ_HF_ID = 'google-research-datasets/natural_questions'


# ---------------------------------------------------------------------------
# Extraction helpers (operate on raw HF example dicts)
# ---------------------------------------------------------------------------

def nq_get_text(doc: dict, strip_html: bool = True) -> str:
    """Extract full document text. Optionally strip HTML tokens."""
    tokens = doc['tokens']['token']
    if strip_html:
        is_html = doc['tokens']['is_html']
        return ' '.join(t for t, h in zip(tokens, is_html) if not h)
    return ' '.join(tokens)


def nq_get_text_span(doc: dict, start_token: int, end_token: int, strip_html: bool = True) -> str:
    """Extract text between token indices. Optionally skip HTML tokens."""
    tokens = doc['tokens']['token']
    if strip_html:
        is_html = doc['tokens']['is_html']
        return ' '.join(
            tokens[i] for i in range(start_token, end_token) if not is_html[i]
        )
    return ' '.join(tokens[start_token:end_token])


def nq_extract_short_answer(example: dict, strip_html: bool = True) -> Optional[str]:
    """Return the first valid short-answer text, or None."""
    doc = example['document']
    ann = example['annotations']
    for sa_list in ann['short_answers']:
        starts = sa_list['start_token']
        ends = sa_list['end_token']
        if len(starts) > 0:
            return nq_get_text_span(doc, starts[0], ends[0], strip_html=strip_html)
    return None


def nq_extract_long_answer_span(example: dict) -> Optional[Tuple[int, int]]:
    """Return (start_token, end_token) of the first valid long answer, or None."""
    ann = example['annotations']
    n_ann = len(ann['id'])
    for j in range(n_ann):
        la = ann['long_answer'][j]
        if la['start_token'] >= 0:
            return la['start_token'], la['end_token']
    return None


def nq_extract_long_answer(example: dict, strip_html: bool = True) -> Optional[str]:
    """Return the first valid long-answer text, or None."""
    span = nq_extract_long_answer_span(example)
    if span is None:
        return None
    return nq_get_text_span(example['document'], span[0], span[1], strip_html=strip_html)


def nq_extract_yes_no(example: dict) -> Optional[str]:
    """Return 'yes' / 'no' if a yes_no_answer is present, else None."""
    for yn in example['annotations']['yes_no_answer']:
        if yn == 0:
            return 'no'
        elif yn == 1:
            return 'yes'
    return None


def nq_is_answerable(example: dict) -> bool:
    """Check if the example has any form of answer."""
    if nq_extract_short_answer(example) is not None:
        return True
    if nq_extract_long_answer_span(example) is not None:
        return True
    if nq_extract_yes_no(example) is not None:
        return True
    return False


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class NaturalQuestionsDataset(QnaBaseDataset):
    """Natural Questions dataset for MixedDecoder training.

    Operates directly on HF Dataset objects — no preprocessing step.
    Context extraction and answer selection happen lazily in `_get_item`.

    For each item, the answer is randomly chosen at `_get_item` time between:
      - short answer (extractive span)
      - long answer (paragraph-level span)
    When both exist, one is picked uniformly at random.
    When only one exists, that one is used.
    For yes/no questions, the yes_no string is used as the answer.
    When no answer exists at all, the global constant `noanswer` is returned.

    Context HTML tokens are randomly included or stripped (coin flip per item).
    Answer HTML tokens are always stripped.
    """

    def __init__(
            self,
            ds: HfDataset,
            tkz: PreTrainedTokenizer,
            inp_len: int,
            max_chunks: int,
            max_ans_toks: int = 100,
            max_prompt_toks: int = 100,
            device=None,
    ):
        super().__init__(
            tkz=tkz, inp_len=inp_len, max_chunks=max_chunks,
            max_ans_toks=max_ans_toks, max_prompt_toks=max_prompt_toks, device=device,
        )
        self.ds = ds
        self.inds = np.arange(len(ds))

    def _get_item(self, idx: int) -> Tuple[str, List[str], List[str], bool]:
        ex = self.ds[idx]
        question = ex['question']['text']

        # Randomly decide whether to keep HTML tokens in context
        ctx_strip_html = bool(np.random.randint(2))
        context = nq_get_text(ex['document'], strip_html=ctx_strip_html)

        # Answers — always strip HTML
        short_ans = nq_extract_short_answer(ex, strip_html=True)
        long_ans = nq_extract_long_answer(ex, strip_html=True)
        yes_no = nq_extract_yes_no(ex)

        answerable = short_ans is not None or long_ans is not None or yes_no is not None
        if not answerable:
            return context, [question], [self.NO_ANSWER_TEXT], False

        # Collect answer candidates and pick one randomly
        candidates: list[str] = []
        if short_ans is not None:
            candidates.append(short_ans)
        if long_ans is not None:
            candidates.append(long_ans)
        if yes_no is not None:
            candidates.append(yes_no)

        answer = candidates[np.random.randint(len(candidates))]
        return context, [question], [answer], True


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_nq(
        cache_dir: str | Path | None = None,
) -> Tuple[HfDataset, HfDataset]:
    """Load NQ train and validation splits from HuggingFace.

    Returns:
        ds_train: HF Dataset for the train split
        ds_val: HF Dataset for the validation split
    """
    kwargs = {}
    if cache_dir is not None:
        kwargs['cache_dir'] = str(cache_dir)
    ds_train = load_dataset(NQ_HF_ID, split='train', **kwargs)
    ds_val = load_dataset(NQ_HF_ID, split='validation', **kwargs)
    print(f'NQ loaded: train={len(ds_train)}, val={len(ds_val)}')
    return ds_train, ds_val
