from typing import Generator, Optional, Union

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from mllm.data.utils import get_squadv2_df, split_df


class QnaItemV2:
    tkz: PreTrainedTokenizer
    ind: int
    context: str
    question: str
    answer: str
    ctx_toks: list[int]
    que_toks: list[int]
    ans_toks: list[int]

    def __init__(
            self, tkz: PreTrainedTokenizer, ind: int, context: str, question: str, answer: str,
    ):
        self.tkz = tkz
        self.ind = ind
        self.context = context
        self.question = question
        self.answer = answer
        self.calc_toks()

    def _tokenize(self, s: str, to_numpy: bool = False) -> Union[list, np.ndarray]:
        toks = self.tkz(s, add_special_tokens=False).input_ids
        if to_numpy:
            toks = np.array(toks)
        return toks

    def calc_toks(self):
        self.ctx_toks = self._tokenize(self.context)
        self.que_toks = self._tokenize(self.question)
        self.ans_toks = self._tokenize(self.answer)


class QnaBatchV2:
    tkz: PreTrainedTokenizer
    items: list[QnaItemV2]
    max_inp_len: int
    max_out_len: int
    device: torch.device
    ctx_toks: np.ndarray
    que_toks: np.ndarray
    ans_toks: np.ndarray
    cq_toks: np.ndarray
    ctx_toks_t: Optional[torch.Tensor] = None
    que_toks_t: Optional[torch.Tensor] = None
    ans_toks_t: Optional[torch.Tensor] = None
    cq_toks_t: Optional[torch.Tensor] = None

    def __init__(self, items: list[QnaItemV2], max_inp_len: int, max_out_len: int, device: Optional[torch.device] = None):
        self.tkz = items[0].tkz
        self.items = items
        self.max_inp_len = max_inp_len
        self.max_out_len = max_out_len
        self.device = device if device is not None else torch.device('cpu')
        self.calc_all()

    def calc_all(self):
        n_batch = len(self.items)

        max_ctx_len, max_que_len, max_ans_len = 0, 0, 0
        for item in self.items:
            max_ctx_len = max(max_ctx_len, len(item.ctx_toks))
            max_que_len = max(max_que_len, len(item.que_toks))
            max_ans_len = max(max_ans_len, len(item.ans_toks))
        max_ctx_len = min(max_ctx_len, self.max_inp_len)
        max_ans_len = min(max_ans_len, self.max_out_len)

        b_ctx_toks = np.full((n_batch, max_ctx_len), self.tkz.pad_token_id)
        b_que_toks = np.full((n_batch, max_que_len), self.tkz.pad_token_id)
        b_ans_toks = np.full((n_batch, max_ans_len), self.tkz.pad_token_id)

        b_c_toks, b_q_toks = [], []
        max_cq_len = 0
        for i, item in enumerate(self.items):
            n_ctx = min(max_ctx_len, len(item.ctx_toks))
            ctx_toks = item.ctx_toks[:n_ctx]
            b_ctx_toks[i, :n_ctx] = ctx_toks
            que_toks = item.que_toks
            n_que = min(max_que_len, len(que_toks))
            b_que_toks[i, :n_que] = item.que_toks[:n_que]
            n_ans = min(max_ans_len, len(item.ans_toks))
            b_ans_toks[i, :n_ans] = item.ans_toks[:n_ans]
            b_c_toks.append(ctx_toks)
            b_q_toks.append(que_toks)
            max_cq_len = max(max_cq_len, len(ctx_toks) + len(que_toks))

        b_cq_toks = np.full((n_batch, max_cq_len + 3), self.tkz.pad_token_id)
        for i in range(len(self.items)):
            c_toks, q_toks = b_c_toks[i], b_q_toks[i]
            nc, nq = len(c_toks), len(q_toks)
            b_cq_toks[i, 0] = self.tkz.cls_token_id
            b_cq_toks[i, 1:nc + 1] = c_toks
            b_cq_toks[i, nc + 1] = self.tkz.sep_token_id
            b_cq_toks[i, nc + 2:nc + 2 + nq] = q_toks
            b_cq_toks[i, nc + 2 + nq] = self.tkz.sep_token_id

        self.ctx_toks = b_ctx_toks
        self.que_toks = b_que_toks
        self.ans_toks = b_ans_toks
        self.cq_toks = b_cq_toks

    def get_tensors(self) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.ctx_toks_t is None:
            self.ctx_toks_t = torch.from_numpy(self.ctx_toks).to(self.device)
            self.que_toks_t = torch.from_numpy(self.que_toks).to(self.device)
            self.ans_toks_t = torch.from_numpy(self.ans_toks).to(self.device)
            self.cq_toks_t = torch.from_numpy(self.cq_toks).to(self.device)
        return self.ctx_toks_t, self.que_toks_t, self.ans_toks_t, self.cq_toks_t


BatchV2It = Generator[QnaBatchV2, None, None]


def get_squadv2_batch_iterator_v2(
        df_sq: pd.DataFrame, tkz: PreTrainedTokenizer, batch_size: int, max_inp_len: int, max_out_len: int, device: torch.device,
) -> BatchV2It:
    inds = np.arange(len(df_sq))
    items = []
    for ind in inds:
        row = df_sq.iloc[ind]
        answers = set(row.answers['text']) or {'-'}
        for answer in answers:
            item = QnaItemV2(tkz=tkz, ind=ind, context=row.context, question=row.question, answer=answer)
            items.append(item)
            if len(items) == batch_size:
                batch = QnaBatchV2(items=items, max_inp_len=max_inp_len, max_out_len=max_out_len, device=device)
                yield batch
                items = []


def get_squadv2_batch_iterators_v2(
        batch_size: int, exclude_empty_answers: bool, tkz: PreTrainedTokenizer, max_inp_len: int, max_out_len: int,
        device: torch.device, val_ratio: float = 0.05,
) -> tuple[BatchV2It, BatchV2It]:
    df_sq = get_squadv2_df(exclude_empty_answers=exclude_empty_answers)
    df_sq_t, df_sq_v = split_df(df_sq, val_ratio=val_ratio)
    print(f'Squad v2 n_total = {len(df_sq)}. n_train = {len(df_sq_t)}. n_val = {len(df_sq_v)}')

    train_batch_it = get_squadv2_batch_iterator_v2(
        df_sq=df_sq_t, tkz=tkz, batch_size=batch_size, max_inp_len=max_inp_len, max_out_len=max_out_len, device=device,
    )
    val_batch_it = get_squadv2_batch_iterator_v2(
        df_sq=df_sq_v, tkz=tkz, batch_size=batch_size, max_inp_len=max_inp_len, max_out_len=max_out_len, device=device,
    )
    return train_batch_it, val_batch_it

