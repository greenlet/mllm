"""NewsQA dataset loader for QnA MixedDecoder training.

HuggingFace: lucadiliello/newsqa
Splits: train (~74K), validation (~4K)
Columns: context, question, answers (list[str]), key, labels
Context: full CNN news article
Answer: answers[0]
Unanswerable: no
Multi-turn: no — multiple questions per article
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HfDataset
from transformers import PreTrainedTokenizer

from mllm.data.qna.dataset import QnaBaseDataset


NEWSQA_HF_ID = 'lucadiliello/newsqa'


class NewsqaDataset(QnaBaseDataset):
    """NewsQA dataset for MixedDecoder training.

    Operates directly on HF Dataset objects — no preprocessing step.
    For each item, the first annotated answer string is used.
    Items with an empty answers list are skipped during index construction.
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
        # Filter out items with empty answers, empty context, or empty question
        self.inds = np.array([
            i for i in range(len(ds))
            if ds[i]['answers'] and ds[i]['context'].strip() and ds[i]['question'].strip()
        ], dtype=np.int64)

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

def load_newsqa(
        cache_dir: str | Path | None = None,
) -> Tuple[HfDataset, HfDataset]:
    """Load NewsQA train and validation splits from HuggingFace.

    Returns:
        ds_train: HF Dataset for the train split
        ds_val: HF Dataset for the validation split
    """
    kwargs = {'trust_remote_code': True}
    if cache_dir is not None:
        kwargs['cache_dir'] = str(cache_dir)
    ds_train = load_dataset(NEWSQA_HF_ID, split='train', **kwargs)
    ds_val = load_dataset(NEWSQA_HF_ID, split='validation', **kwargs)
    print(f'NewsQA loaded: train={len(ds_train)}, val={len(ds_val)}')
    return ds_train, ds_val
