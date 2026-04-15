"""MRQA dataset loader for QnA MixedDecoder training.

HuggingFace: mrqa
Splits: train (~517K), validation (~58K), test (~10K)
Columns: subset, context, question, answers (list[str]), detected_answers
Train subsets: SQuAD, NewsQA, TriviaQA-web, SearchQA, HotpotQA, NaturalQuestionsShort
Only SearchQA and HotpotQA subsets are kept to avoid overlap with standalone datasets.
Unanswerable: no
Multi-turn: no
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HfDataset
from transformers import PreTrainedTokenizer

from mllm.data.qna.dataset import QnaBaseDataset


MRQA_HF_ID = 'mrqa'
MRQA_KEEP_SUBSETS = {'SearchQA', 'HotpotQA'}


class MrqaDataset(QnaBaseDataset):
    """MRQA dataset (SearchQA + HotpotQA subsets) for MixedDecoder training.

    Operates directly on HF Dataset objects pre-filtered to the kept subsets.
    For each item, one of the annotated answer strings is chosen at random.
    All items are answerable (no empty answers in these subsets).
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
        context = ex['context']
        question = ex['question']

        answer_texts: List[str] = ex['answers']
        if len(answer_texts) == 0:
            return context, [question], [self.NO_ANSWER_TEXT], False

        answer = answer_texts[np.random.randint(len(answer_texts))]
        return context, [question], [answer], True


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _filter_subsets(ds: HfDataset, keep: set[str]) -> HfDataset:
    """Filter an MRQA split to keep only rows whose subset is in *keep*."""
    return ds.filter(lambda ex: ex['subset'] in keep)


def load_mrqa(
        cache_dir: str | Path | None = None,
) -> Tuple[HfDataset, HfDataset]:
    """Load MRQA train and validation splits, filtered to SearchQA + HotpotQA.

    Returns:
        ds_train: HF Dataset for the filtered train split
        ds_val: HF Dataset for the filtered validation split
    """
    kwargs = {}
    if cache_dir is not None:
        kwargs['cache_dir'] = str(cache_dir)
    ds_train = load_dataset(MRQA_HF_ID, split='train', **kwargs)
    ds_val = load_dataset(MRQA_HF_ID, split='validation', **kwargs)
    ds_train = _filter_subsets(ds_train, MRQA_KEEP_SUBSETS)
    ds_val = _filter_subsets(ds_val, MRQA_KEEP_SUBSETS)
    print(f'MRQA loaded (SearchQA + HotpotQA): train={len(ds_train)}, val={len(ds_val)}')
    return ds_train, ds_val

