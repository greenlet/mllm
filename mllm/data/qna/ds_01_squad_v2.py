"""SQuAD v2 dataset loader for QnA MixedDecoder training.

HuggingFace: rajpurkar/squad_v2
Splits: train (~130K), validation (~11.8K)
Columns: id, title, context, question, answers (text: list[str], answer_start: list[int])
Context: paragraph text
Answer: one of the annotated answer spans chosen at random
Unanswerable: yes — answers['text'] is empty (~33% of data) → answer = 'noanswer'
Multi-turn: no
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HfDataset
from transformers import PreTrainedTokenizer

from mllm.data.qna.dataset import QnaBaseDataset


SQUAD_V2_HF_ID = 'rajpurkar/squad_v2'


class SquadV2Dataset(QnaBaseDataset):
    """SQuAD v2 dataset for MixedDecoder training.

    Operates directly on HF Dataset objects — no preprocessing step.
    For answerable items with multiple answer annotations, one is picked
    uniformly at random in `_get_item`.
    For unanswerable items (empty answers['text']), the constant `noanswer`
    is returned — same convention as Natural Questions.
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

        answer_texts: List[str] = ex['answers']['text']
        if len(answer_texts) == 0:
            # Unanswerable — return constant, same as NQ
            return context, [question], [self.NO_ANSWER_TEXT], False

        # Pick a random answer span from annotations
        answer = answer_texts[np.random.randint(len(answer_texts))]
        return context, [question], [answer], True


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_squad_v2(
        cache_dir: str | Path | None = None,
) -> Tuple[HfDataset, HfDataset]:
    """Load SQuAD v2 train and validation splits from HuggingFace.

    Returns:
        ds_train: HF Dataset for the train split
        ds_val: HF Dataset for the validation split
    """
    kwargs = {}
    if cache_dir is not None:
        kwargs['cache_dir'] = str(cache_dir)
    ds_train = load_dataset(SQUAD_V2_HF_ID, split='train', **kwargs)
    ds_val = load_dataset(SQUAD_V2_HF_ID, split='validation', **kwargs)
    print(f'SQuAD v2 loaded: train={len(ds_train)}, val={len(ds_val)}')
    return ds_train, ds_val
