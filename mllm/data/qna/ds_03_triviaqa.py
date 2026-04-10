"""TriviaQA dataset loader for QnA MixedDecoder training.

HuggingFace: trivia_qa (config: rc)
Splits: train (~138K), validation (~18K), test (~18K)
Columns: question, question_id, answer (dict: value, aliases, normalized_aliases),
         entity_pages (title, wiki_context), search_results (title, url, search_context)
Context: random subset of wiki_context and search_context sources, concatenated in random order
Answer: randomly chosen from answer aliases (includes canonical value)
Unanswerable: no
Multi-turn: no
"""

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HfDataset
from transformers import PreTrainedTokenizer

from mllm.data.qna.dataset import QnaBaseDataset


TRIVIAQA_HF_ID = 'trivia_qa'
TRIVIAQA_SUBSET = 'rc'


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into a single space."""
    return re.sub(r'[\s\r\n]+', ' ', text).strip()


def _collect_sources(example: dict) -> List[str]:
    """Return all non-empty evidence passages (wiki + search) for an example."""
    sources: List[str] = []
    for ctx in example['entity_pages']['wiki_context']:
        ctx = _normalize_whitespace(ctx)
        if ctx:
            sources.append(ctx)
    for ctx in example['search_results']['search_context']:
        ctx = _normalize_whitespace(ctx)
        if ctx:
            sources.append(ctx)
    return sources


def _build_context(sources: List[str], rng: np.random.Generator | None = None) -> str:
    """Pick a random non-empty subset of *sources* in random order and concatenate.

    At least 1 source is always included (if available).
    """
    if not sources:
        return ''
    if rng is None:
        n = np.random.randint(1, len(sources) + 1)
        chosen_idx = np.random.choice(len(sources), size=n, replace=False)
    else:
        n = rng.integers(1, len(sources) + 1)
        chosen_idx = rng.choice(len(sources), size=n, replace=False)
    return ' '.join(sources[i] for i in chosen_idx)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class TriviaQADataset(QnaBaseDataset):
    """TriviaQA dataset for MixedDecoder training.

    Operates directly on HF Dataset objects — no preprocessing step.
    Context is built lazily in `_get_item` by randomly selecting and ordering
    a subset of all available evidence sources (Wikipedia pages + web search
    results).  The answer is randomly chosen from the answer aliases list.
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
        question = ex['question']

        # Build context from a random subset of all evidence sources
        sources = _collect_sources(ex)
        context = _build_context(sources)

        # Pick a random answer alias
        aliases: List[str] = ex['answer']['aliases']
        if len(aliases) > 0:
            answer = aliases[np.random.randint(len(aliases))]
        else:
            # Fallback to canonical value (should always exist)
            answer = ex['answer']['value']

        return context, [question], [answer], True


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_triviaqa(
        cache_dir: str | Path | None = None,
) -> Tuple[HfDataset, HfDataset]:
    """Load TriviaQA RC train and validation splits from HuggingFace.

    Returns:
        ds_train: HF Dataset for the train split
        ds_val: HF Dataset for the validation split
    """
    kwargs = {}
    if cache_dir is not None:
        kwargs['cache_dir'] = str(cache_dir)
    ds_train = load_dataset(TRIVIAQA_HF_ID, TRIVIAQA_SUBSET, split='train', **kwargs)
    ds_val = load_dataset(TRIVIAQA_HF_ID, TRIVIAQA_SUBSET, split='validation', **kwargs)
    print(f'TriviaQA loaded: train={len(ds_train)}, val={len(ds_val)}')
    return ds_train, ds_val
