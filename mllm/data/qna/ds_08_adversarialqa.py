"""AdversarialQA dataset loader for QnA MixedDecoder training.

HuggingFace: adversarial_qa (config: adversarialQA)
Splits: train (~30K), validation (~3K), test (~3K)
Columns: id, title, context, question, answers (text: list[str], answer_start: list[int])
SQuAD-identical format. Adversarially collected to be hard but answerable.
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


ADVERSARIALQA_HF_ID = 'adversarial_qa'
ADVERSARIALQA_SUBSET = 'adversarialQA'


class AdversarialqaDataset(QnaBaseDataset):
    """AdversarialQA dataset for MixedDecoder training.

    Operates directly on HF Dataset objects — no preprocessing step.
    SQuAD-identical schema: answers['text'] contains annotated spans.
    All items are answerable; one answer is picked at random when multiple exist.
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
            return context, [question], [self.NO_ANSWER_TEXT], False

        answer = answer_texts[np.random.randint(len(answer_texts))]
        return context, [question], [answer], True


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_adversarialqa(
        cache_dir: str | Path | None = None,
) -> Tuple[HfDataset, HfDataset]:
    """Load AdversarialQA train and validation splits from HuggingFace.

    Returns:
        ds_train: HF Dataset for the train split
        ds_val: HF Dataset for the validation split
    """
    kwargs = {'trust_remote_code': True}
    if cache_dir is not None:
        kwargs['cache_dir'] = str(cache_dir)
    ds_train = load_dataset(ADVERSARIALQA_HF_ID, ADVERSARIALQA_SUBSET, split='train', **kwargs)
    ds_val = load_dataset(ADVERSARIALQA_HF_ID, ADVERSARIALQA_SUBSET, split='validation', **kwargs)
    print(f'AdversarialQA loaded: train={len(ds_train)}, val={len(ds_val)}')
    return ds_train, ds_val

