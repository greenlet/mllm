"""CoQA dataset loader for QnA MixedDecoder training.

HuggingFace: stanfordnlp/coqa
Splits: train, validation
Columns: source (domain), story, questions (list[str]),
         answers (input_text, answer_start, answer_end per turn)
Context: story (passage from 5 domains: race, cnn, wikipedia, gutenberg, mctest)
Answer: answers['input_text'] — abstractive human-written answer (preferred over extractive span)
Unanswerable: yes — input_text == 'unknown' (~1.3% of turns)
Multi-turn: yes — each row is a multi-turn dialogue
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HfDataset
from transformers import PreTrainedTokenizer

from mllm.data.qna.dataset import QnaBaseDataset


COQA_HF_ID = 'stanfordnlp/coqa'
COQA_UNKNOWN = 'unknown'


class CoqaDataset(QnaBaseDataset):
    """CoQA (multi-turn conversational QA) dataset for MixedDecoder training.

    Each HF row is a full dialogue over a story passage.  At training time
    a random turn is chosen and the prompt is built from the conversation
    history up to that turn:  Q: q0 A: a0. Q: q1 A: a1. ... Q: q_i A:

    Answers are the abstractive ``input_text`` strings written by annotators.
    Turns where ``input_text`` is ``'unknown'`` are treated as unanswerable.
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
        context = ex['story']

        all_questions: List[str] = ex['questions']
        all_answers: List[str] = ex['answers']['input_text']
        n_turns = len(all_questions)

        # Pick a random turn as the target
        i_turn = np.random.randint(n_turns)

        # Questions: history + current
        questions = all_questions[:i_turn + 1]
        # Answers: history + target (parallel with questions)
        answers = all_answers[:i_turn + 1]

        target = answers[-1]
        is_answerable = target.lower().strip() != COQA_UNKNOWN
        if not is_answerable:
            answers = list(answers)
            answers[-1] = self.NO_ANSWER_TEXT

        return context, questions, answers, is_answerable


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_coqa(
        cache_dir: str | Path | None = None,
) -> Tuple[HfDataset, HfDataset]:
    """Load CoQA train and validation splits from HuggingFace.

    Returns:
        ds_train: HF Dataset for the train split
        ds_val: HF Dataset for the validation split
    """
    kwargs = {'trust_remote_code': True}
    if cache_dir is not None:
        kwargs['cache_dir'] = str(cache_dir)
    ds_train = load_dataset(COQA_HF_ID, split='train', **kwargs)
    ds_val = load_dataset(COQA_HF_ID, split='validation', **kwargs)
    n_train_turns = sum(len(ds_train[i]['questions']) for i in range(len(ds_train)))
    n_val_turns = sum(len(ds_val[i]['questions']) for i in range(len(ds_val)))
    print(f'CoQA loaded: train={len(ds_train)} dialogues ({n_train_turns} turns), '
          f'val={len(ds_val)} dialogues ({n_val_turns} turns)')
    return ds_train, ds_val
