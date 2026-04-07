"""SQuAD v1.1 dataset loader for QnA MixedDecoder training.

HuggingFace: rajpurkar/squad
Splits: train (~87K), validation (~10K)
Columns: id, title, context, question, answers (text: list[str], answer_start: list[int])
Same format as SQuAD v2 but answerable-only.
Needs deduplication against SQuAD v2 by id (v2 contains all v1 answerable questions).
Context: paragraph text
Answer: one of the annotated answer spans chosen at random
Unanswerable: no
Multi-turn: no
"""

from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HfDataset
from transformers import PreTrainedTokenizer

from mllm.data.qna.dataset import QnaBaseDataset


SQUAD_V1_HF_ID = 'rajpurkar/squad'


class SquadV1Dataset(QnaBaseDataset):
    """SQuAD v1.1 dataset for MixedDecoder training.

    Operates directly on HF Dataset objects — no preprocessing step.
    All items are answerable. For items with multiple answer annotations,
    one is picked uniformly at random in `_get_item`.
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

    def _get_item(self, idx: int) -> Tuple[str, str, str, bool]:
        ex = self.ds[idx]
        context = ex['context']
        question = ex['question']

        answer_texts: List[str] = ex['answers']['text']
        # SQuAD v1 is answerable-only; every item has at least one answer
        answer = answer_texts[np.random.randint(len(answer_texts))]
        return context, question, answer, True


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_squad_v1(
        cache_dir: str | Path | None = None,
) -> Tuple[HfDataset, HfDataset]:
    """Load SQuAD v1.1 train and validation splits from HuggingFace.

    Returns:
        ds_train: HF Dataset for the train split
        ds_val: HF Dataset for the validation split
    """
    kwargs = {}
    if cache_dir is not None:
        kwargs['cache_dir'] = str(cache_dir)
    ds_train = load_dataset(SQUAD_V1_HF_ID, split='train', **kwargs)
    ds_val = load_dataset(SQUAD_V1_HF_ID, split='validation', **kwargs)
    print(f'SQuAD v1.1 loaded: train={len(ds_train)}, val={len(ds_val)}')
    return ds_train, ds_val


def get_squad_v2_ids(ds_v2: HfDataset) -> Set[str]:
    """Return the set of all example IDs in a SQuAD v2 split.

    Useful for deduplicating SQuAD v1 when training with both versions,
    since v2 contains all v1 answerable questions.
    """
    return set(ds_v2['id'])
