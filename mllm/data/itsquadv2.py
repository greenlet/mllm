from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset

from transformers import PreTrainedTokenizer

from mllm.data.utils import get_squadv2_df, split_df



class QnaBatchV2:
    qas: list[tuple[str, str]]
    contexts: list[str]
    toks_seq_len: int
    tkz: PreTrainedTokenizer
    ques_inp: QnaQuesInp
    q_toks: list[np.ndarray]
    a_toks: list[np.ndarray]
    qa_toks: list[np.ndarray]
    a_att_masks: list[np.ndarray]
    a_tgt_masks: list[np.ndarray]
    qa_att_masks: list[np.ndarray]
    qa_tgt_masks: list[np.ndarray]
    ctx_toks: np.ndarray
    qa_toks_t: list[torch.Tensor]
    q_toks_t: list[torch.Tensor]
    a_toks_t: list[torch.Tensor]
    a_att_masks_t: list[torch.Tensor]
    a_tgt_masks_t: list[torch.Tensor]
    qa_att_mask_t: list[torch.Tensor]
    qa_tgt_mask_t: list[torch.Tensor]
    device: Optional[torch.device] = None
    ctx_toks_t: Optional[torch.Tensor] = None

    def __init__(
            self, qas: list[tuple[str, str]], contexts: list[str], toks_seq_len: int, tkz: PreTrainedTokenizer,
            ques_inp: QnaQuesInp, device: Optional[torch.device] = None,
    ):
        self.qas = qas
        self.contexts = contexts
        self.toks_seq_len = toks_seq_len
        self.tkz = tkz
        self.ques_inp = ques_inp
        self.device = device
        self._process()

    def _process(self):
        # Question + Answer
        q_toks_l, a_toks_l, qa_toks_l, a_att_masks_l, a_tgt_masks_l, qa_att_masks_l, qa_tgt_masks_l = [], [], [], [], [], [], []
        qas_cum, qas_sq_cum, as_cum = 0, 0, 0
        for q, a in self.qas:
            q = f'Question: {q}. Answer: '
            q_toks: list[int] = self.tkz(q).input_ids
            a_toks: list[int] = self.tkz(a).input_ids
            assert q_toks[0] == a_toks[0] == self.tkz.cls_token_id, f'q_toks[0] = {q_toks[0]}. a_toks[0] = {a_toks[0]}'
            assert q_toks[-1] == a_toks[-1] == self.tkz.sep_token_id, f'q_toks[-1] = {q_toks[-1]}. a_toks[-1] = {a_toks[-1]}'
            q_toks, a_toks = q_toks[0:-1], a_toks[1:]

            # # TODO: parametrize
            # if self.ques_inp == QnaQuesInp.Dec:
            #     if len(a_toks) > 18:
            #         a_toks = a_toks[:17] + a_toks[-1:]
            #     if len(a_toks) > 10 and len(q_toks) + len(a_toks) > 30:
            #         if len(q_toks) > 20:
            #             q_toks = q_toks[:1] + q_toks[-19:]
            #         a_toks = a_toks[:9] + a_toks[-1:]
            # elif self.ques_inp == QnaQuesInp.Enc and len(a_toks) > 20:
            #     a_toks = a_toks[:-19] + a_toks[-1:]

            assert len(a_toks) > 1 and a_toks[0] != self.tkz.cls_token_id and a_toks[-1] == self.tkz.sep_token_id, \
                f'a_toks must contain at least one content token and SEP token at the end. a_toks = {a_toks}'
            assert len(q_toks) > 1 and q_toks[0] == self.tkz.cls_token_id and q_toks[-1] != self.tkz.sep_token_id, \
                f'q_toks must contain at least one content token and CLS token at the start. q_toks = {q_toks}'


            qa_toks = [*q_toks, self.tkz.sep_token_id, *a_toks]

            # # TODO: parametrize
            # qa_len = len(qa_toks)
            # qa_len_sq = qa_len**2
            # a_len = len(a_toks)
            # if self.ques_inp == QnaQuesInp.Dec and \
            #         (qas_sq_cum + qa_len_sq >= 2800 or as_cum + a_len > 25 or qas_cum + qa_len > 10000):
            #     continue
            # if self.ques_inp == QnaQuesInp.Enc and as_cum + a_len > 30:
            #     continue
            # qas_cum += qa_len
            # qas_sq_cum += qa_len_sq
            # as_cum += a_len

            if self.ques_inp == QnaQuesInp.Enc:
                assert q_toks[-1] != self.tkz.sep_token_id
                n_toks = len(q_toks)
                if n_toks > self.toks_seq_len:
                    q_toks = q_toks[:self.toks_seq_len]
                elif n_toks < self.toks_seq_len:
                    n_pad = self.toks_seq_len - n_toks
                    q_toks += [self.tkz.pad_token_id] * n_pad

            q_toks_l.append(np.array(q_toks, dtype=int))
            a_toks_l.append(np.array(a_toks, dtype=int))
            qa_toks_l.append(np.array(qa_toks, dtype=int))

            n_q_toks, n_a_toks = len(q_toks), len(a_toks)
            q_mask = np.ones((n_a_toks, n_q_toks + 1), dtype=int)
            a_att_mask = np.ones((n_a_toks, n_a_toks), dtype=int)
            a_att_mask = np.tril(a_att_mask, k=0)
            a_tgt_mask = np.eye(n_a_toks, dtype=int)
            qa_att_mask = np.concatenate([q_mask, a_att_mask], axis=1)
            qa_tgt_mask = np.concatenate([q_mask * 0, a_tgt_mask], axis=1).astype(bool)
            a_att_masks_l.append(a_att_mask)
            a_tgt_masks_l.append(a_tgt_mask.astype(bool))
            qa_att_masks_l.append(qa_att_mask)
            qa_tgt_masks_l.append(qa_tgt_mask)
        self.q_toks = q_toks_l
        self.a_toks = a_toks_l
        self.qa_toks = qa_toks_l
        self.a_att_masks = a_att_masks_l
        self.a_tgt_masks = a_tgt_masks_l
        self.qa_att_masks = qa_att_masks_l
        self.qa_tgt_masks = qa_tgt_masks_l

        # Contexts
        ctxs = []
        max_ctx_chunks = 3
        for ctx in self.contexts:
            ctx_toks = self.tkz(ctx).input_ids
            assert ctx_toks[0] == self.tkz.cls_token_id, f'ctx_token[0] (={ctx_toks[0]}) != cls_token_id (={self.tkz.cls_token_id})'
            assert ctx_toks[-1] == self.tkz.sep_token_id, f'ctx_token[-1] (={ctx_toks[-1]}) != sep_token_id (={self.tkz.sep_token_id})'
            ctx_toks = ctx_toks[:-1]
            n_pad = self.toks_seq_len - len(ctx_toks) % self.toks_seq_len
            assert self.tkz.pad_token_id is not None
            ctx_toks = np.pad(ctx_toks, (0, n_pad), constant_values=self.tkz.pad_token_id)
            ctx_toks = ctx_toks[:self.toks_seq_len][None]
            ctxs.append(ctx_toks[:max_ctx_chunks])
        ctxs_all = np.concatenate(ctxs)
        self.ctx_toks = ctxs_all

        # ctxs_lens = np.array([len(c) for c in ctxs])
        # qas_lens = np.array([len(qa) for qa in self.qa_toks])
        # qs_lens = np.array([len(q) for q in self.q_toks])
        # as_lens = np.array([len(a) for a in self.a_toks])
        # print(f'Contexts: {ctxs_lens}. {ctxs_all.shape}')
        # print(f'QAs: {qas_lens}. {qas_lens.sum()}. {np.square(qas_lens).sum()}')
        # print(f'Qs: {qs_lens}. {qs_lens.sum()}. {np.square(qs_lens).sum()}')
        # print(f'As: {as_lens}. {as_lens.sum()}. {np.square(as_lens).sum()}')

    def _to_tensor_single(self, arr: np.ndarray) -> torch.Tensor:
        res = torch.from_numpy(arr)
        if self.device is not None:
            res = res.to(self.device)
        return res

    def _to_tensor_multi(self, arr: list[np.ndarray]) -> list[torch.Tensor]:
        return [self._to_tensor_single(x) for x in arr]

    def gen_tensors(self) -> tuple[torch.Tensor, tuple[list[torch.Tensor], ...]]:
        if self.ques_inp == QnaQuesInp.Enc:
            if self.ctx_toks_t is None:
                self.q_toks_t = self._to_tensor_multi(self.q_toks)
                self.a_toks_t = self._to_tensor_multi(self.a_toks)
                self.a_att_masks_t = self._to_tensor_multi(self.a_att_masks)
                self.a_tgt_masks_t = self._to_tensor_multi(self.a_tgt_masks)
                self.ctx_toks_t = self._to_tensor_single(self.ctx_toks)
            return self.ctx_toks_t, (self.q_toks_t, self.a_toks_t, self.a_att_masks_t, self.a_tgt_masks_t)

        if self.ques_inp == QnaQuesInp.Dec:
            if self.ctx_toks_t is None:
                self.qa_toks_t = self._to_tensor_multi(self.qa_toks)
                self.qa_att_mask_t = self._to_tensor_multi(self.qa_att_masks)
                self.qa_tgt_mask_t = self._to_tensor_multi(self.qa_tgt_masks)
                self.ctx_toks_t = self._to_tensor_single(self.ctx_toks)
            return self.ctx_toks_t, (self.qa_toks_t, self.qa_att_mask_t, self.qa_tgt_mask_t)

        raise Exception(f'Question input type {self.ques_inp} is not supported')


# df_sq: ['id', 'title', 'context', 'question', 'answers']
def get_squadv2_batch(tkz: PreTrainedTokenizer, df_sq: pd.DataFrame, inds: np.ndarray, inp_len: int, device: torch.device, ques_inp: QnaQuesInp) -> QnaBatchV2:
    df_b = df_sq.iloc[inds]
    ctxs, ctx_num, qas = {}, 0, set()
    for _, row in df_b.iterrows():
        if not row.context in ctxs:
            ctx_num += 1
            ctxs[row.context] = f'Context{ctx_num}'
        answers = row.answers['text']
        if len(answers) == 0:
            answers = ['-']
        for answer in answers:
            # q = f'{ctxs[row.context]}. Question: {row.question}'
            # q = f'Question: {row.question}. Answer: '
            q = row.question
            qa = q, answer
            qas.add(qa)
    contexts = [f'{val}. {key}' for key, val in ctxs.items()]
    qas = list(qas)
    n_qas, n_batch = len(qas), len(df_b)
    # max_sz = 2
    max_sz = len(contexts)
    if n_qas > max_sz:
        np.random.shuffle(qas)
        qas = qas[:max_sz]
    return QnaBatchV2(qas=qas, contexts=contexts, toks_seq_len=inp_len, tkz=tkz, device=device, ques_inp=ques_inp)


BatchV2It = Generator[QnaBatchV2, None, None]


def get_squadv2_batch_iterator_v2(
        df_sq: pd.DataFrame, tkz: PreTrainedTokenizer, batch_size: int, inp_len: int, device: torch.device,
) -> BatchV2It:
    inds = np.arange(len(df_sq))
    batch_off = 0
    while True:
        batch_inds = inds[batch_off:batch_off + batch_size]
        n_cur = len(batch_inds)
        n_rest = batch_size - n_cur
        if n_rest > 0:
            batch_inds = np.concatenate([batch_inds, inds[:n_rest]])
        sq_batch = get_squadv2_batch(tkz=tkz, df_sq=df_sq, inds=batch_inds, inp_len=inp_len, device=device)
        yield sq_batch
        batch_off += batch_size
        if batch_off >= len(inds):
            batch_off = 0
            np.random.shuffle(inds)


def get_squadv2_tensor_iterators_v2(
        inp_len: int, batch_size: int, exclude_empty_answers: bool, tkz: PreTrainedTokenizer,
        device: torch.device, val_ratio: float = 0.05,
) -> tuple[BatchV2It, BatchV2It]:
    df_sq = get_squadv2_df(exclude_empty_answers=exclude_empty_answers)
    df_sq_t, df_sq_v = split_df(df_sq, val_ratio=val_ratio)
    print(f'Squad v2 n_total = {len(df_sq)}. n_train = {len(df_sq_t)}. n_val = {len(df_sq_v)}')

    train_batch_it = get_squadv2_batch_iterator_v2(
        df_sq=df_sq_t, tkz=tkz, batch_size=batch_size, inp_len=inp_len, device=device,
    )
    val_batch_it = get_squadv2_batch_iterator_v2(
        df_sq=df_sq_v, tkz=tkz, batch_size=batch_size, inp_len=inp_len, device=device,
    )
    return train_batch_it, val_batch_it

