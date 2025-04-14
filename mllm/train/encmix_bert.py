from dataclasses import dataclass
from typing import Generator

import numpy as np
import pandas as pd

from mllm.train.utils import get_squadv2_df, split_df


@dataclass
class QnaTuple:
    ind: int
    context: str
    question: str
    answer: str

QnaTxtGen = Generator[QnaTuple, None, None]


def get_squadv2_txt_iterator(df_squad: pd.DataFrame) -> QnaTxtGen:
    n = len(df_squad)
    inds = np.arange(n)
    i_off = 0
    while True:
        ind = inds[i_off].item()
        row = df_squad.iloc[ind]
        context, question, answers = row['context'], row['question'], row['answers']['text']
        if len(answers) == 0:
            answers = ['-']

        for answer in answers:
            qna_tuple = QnaTuple(ind=ind, context=context, question=question, answer=answer)
            yield qna_tuple

        if i_off == n:
            np.random.shuffle(inds)
            i_off = 0


def get_squadv2_txt_iterators(exclude_empty_answers: bool, val_ratio: float = 0.05) -> tuple[QnaTxtGen, QnaTxtGen]:
    df_sq = get_squadv2_df(exclude_empty_answers=exclude_empty_answers)
    df_sq_t, df_sq_v = split_df(df_sq, val_ratio=val_ratio)
    print(f'Squad v2 n_total = {len(df_sq)}. n_train = {len(df_sq_t)}. n_val = {len(df_sq_v)}')

    train_it = get_squadv2_txt_iterator(
        df_squad=df_sq_t,
    )
    val_it = get_squadv2_txt_iterator(
        df_squad=df_sq_v,
    )
    return train_it, val_it

