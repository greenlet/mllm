"""QuAC dataset loader for QnA MixedDecoder training.

HuggingFace: quac (revision: refs/convert/parquet)
Splits: train, validation
Columns: dialogue_id, wikipedia_page_title, background, section_title, context,
         questions (list[str]), orig_answers (texts, answer_starts), followups, yesnos, turn_ids
Context: Wikipedia section passage shared across all turns
Answer: orig_answers['texts'] per turn; 'CANNOTANSWER' for unanswerable turns (~17%)
Unanswerable: yes
Multi-turn: yes — each row is a full dialogue with multiple Q/A turns
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HfDataset
from transformers import PreTrainedTokenizer

from mllm.data.qna.dataset import QnaBaseDataset


QUAC_HF_ID = 'quac'
QUAC_CANNOTANSWER = 'CANNOTANSWER'


class QuacDataset(QnaBaseDataset):
    """QuAC (multi-turn conversational QA) dataset for MixedDecoder training.

    Each HF row is a full dialogue.  At training time a random turn is
    chosen and the prompt is built from the conversation history up to
    that turn:  Q: q0 A: a0. Q: q1 A: a1. ... Q: q_i A:

    CANNOTANSWER turns are included — the model learns to predict
    the unanswerable token just like SQuAD v2.
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

        all_questions: List[str] = ex['questions']
        all_answers: List[str] = ex['orig_answers']['texts']
        n_turns = len(all_questions)

        # Pick a random turn as the target
        i_turn = np.random.randint(n_turns)

        # Questions: history + current
        questions = [*all_questions[:i_turn + 1]]
        wikipedia_page_title = ex['wikipedia_page_title']
        questions[-1] = f'({wikipedia_page_title}) {questions[-1]}'
        # Answers: history + target (parallel with questions)
        answers = all_answers[:i_turn + 1]

        target = answers[-1]
        is_answerable = target != QUAC_CANNOTANSWER
        if not is_answerable:
            answers = list(answers)
            answers[-1] = self.NO_ANSWER_TEXT

        return context, questions, answers, is_answerable


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_quac(
        cache_dir: str | Path | None = None,
) -> Tuple[HfDataset, HfDataset]:
    """Load QuAC train and validation splits from HuggingFace.

    Returns:
        ds_train: HF Dataset for the train split
        ds_val: HF Dataset for the validation split
    """
    kwargs = {}
    if cache_dir is not None:
        kwargs['cache_dir'] = str(cache_dir)
    ds_train = load_dataset(QUAC_HF_ID, split='train', revision='refs/convert/parquet', **kwargs)
    ds_val = load_dataset(QUAC_HF_ID, split='validation', revision='refs/convert/parquet', **kwargs)        
    n_train_turns = sum(len(ds_train[i]['questions']) for i in range(len(ds_train)))
    n_val_turns = sum(len(ds_val[i]['questions']) for i in range(len(ds_val)))
    print(f'QuAC loaded: train={len(ds_train)} dialogues ({n_train_turns} turns), '
          f'val={len(ds_val)} dialogues ({n_val_turns} turns)')
    return ds_train, ds_val
