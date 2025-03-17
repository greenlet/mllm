from enum import Enum
from typing import Optional, Generator, Union

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions

from mllm.model.embgen_bert import EncoderEmbDecoderModel


class QuesInp(str, Enum):
    Enc = 'enc'
    Dec = 'dec'


class QnaBatch:
    qas: list[tuple[str, str]]
    contexts: list[str]
    toks_seq_len: int
    tkz: PreTrainedTokenizer
    ques_inp: QuesInp
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
            ques_inp: QuesInp, device: Optional[torch.device] = None,
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
        qas_sq_cum, as_cum = 0, 0
        for q, a in self.qas:
            q_toks: list[int] = self.tkz(q).input_ids
            a_toks: list[int] = self.tkz(a).input_ids
            assert q_toks[0] == a_toks[0] == self.tkz.cls_token_id, f'q_toks[0] = {q_toks[0]}. a_toks[0] = {a_toks[0]}'
            assert q_toks[-1] == a_toks[-1] == self.tkz.sep_token_id, f'q_toks[-1] = {q_toks[-1]}. a_toks[-1] = {a_toks[-1]}'
            q_toks, a_toks = q_toks[0:-1], a_toks[1:]

            # TODO: parametrize
            if self.ques_inp == QuesInp.Dec and len(a_toks) > 18:
                a_toks = a_toks[-18:]
            elif self.ques_inp == QuesInp.Enc and len(a_toks) > 20:
                a_toks = a_toks[-20:]

            qa_toks = [*q_toks, self.tkz.sep_token_id, *a_toks]
            qa_len_sq = len(qa_toks)**2
            a_len = len(a_toks)

            # TODO: parametrize
            if self.ques_inp == QuesInp.Dec and \
                    (qas_sq_cum + qa_len_sq >= 2900 or as_cum + a_len > 20 or len(qa_toks) > 350):
                continue
            if self.ques_inp == QuesInp.Enc and as_cum + a_len > 30:
                continue

            qas_sq_cum += qa_len_sq
            as_cum += a_len

            if self.ques_inp == QuesInp.Enc:
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

        ctxs_lens = np.array([len(c) for c in ctxs])
        qas_lens = np.array([len(qa) for qa in self.qa_toks])
        qs_lens = np.array([len(q) for q in self.q_toks])
        as_lens = np.array([len(a) for a in self.a_toks])
        print(f'Contexts: {ctxs_lens}. {ctxs_all.shape}')
        print(f'QAs: {qas_lens}. {qas_lens.sum()}. {np.square(qas_lens).sum()}')
        print(f'Qs: {qs_lens}. {qs_lens.sum()}. {np.square(qs_lens).sum()}')
        print(f'As: {as_lens}. {as_lens.sum()}. {np.square(as_lens).sum()}')

    def _to_tensor_single(self, arr: np.ndarray) -> torch.Tensor:
        res = torch.from_numpy(arr)
        if self.device is not None:
            res = res.to(self.device)
        return res

    def _to_tensor_multi(self, arr: list[np.ndarray]) -> list[torch.Tensor]:
        return [self._to_tensor_single(x) for x in arr]

    def gen_tensors(self) -> tuple[torch.Tensor, tuple[list[torch.Tensor], ...]]:
        if self.ques_inp == QuesInp.Enc:
            if self.ctx_toks_t is None:
                self.q_toks_t = self._to_tensor_multi(self.q_toks)
                self.a_toks_t = self._to_tensor_multi(self.a_toks)
                self.a_att_masks_t = self._to_tensor_multi(self.a_att_masks)
                self.a_tgt_masks_t = self._to_tensor_multi(self.a_tgt_masks)
                self.ctx_toks_t = self._to_tensor_single(self.ctx_toks)
            return self.ctx_toks_t, (self.q_toks_t, self.a_toks_t, self.a_att_masks_t, self.a_tgt_masks_t)

        if self.ques_inp == QuesInp.Dec:
            if self.ctx_toks_t is None:
                self.qa_toks_t = self._to_tensor_multi(self.qa_toks)
                self.qa_att_mask_t = self._to_tensor_multi(self.qa_att_masks)
                self.qa_tgt_mask_t = self._to_tensor_multi(self.qa_tgt_masks)
                self.ctx_toks_t = self._to_tensor_single(self.ctx_toks)
            return self.ctx_toks_t, (self.qa_toks_t, self.qa_att_mask_t, self.qa_tgt_mask_t)

        raise Exception(f'Question input type {self.ques_inp} is not supported')


def get_sq_df(exclude_empty_answers: bool = False) -> pd.DataFrame:
    ds_name = 'squad_v2'
    ds_sq = load_dataset(ds_name)
    df_sq = pd.concat([ds_sq['train'].to_pandas(), ds_sq['validation'].to_pandas()], axis=0)
    n_total = len(df_sq)
    df_sq = df_sq.sample(n_total)
    if exclude_empty_answers:
        mask = df_sq['answers'].apply(lambda ans: len(ans['text']) > 0)
        df_sq = df_sq[mask]
        print(f'Remove empty answers from dataset {ds_name}. Size: {n_total} --> {len(df_sq)}')
    return df_sq


def split_df(df: pd.DataFrame, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_total = len(df)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    return df.iloc[:n_train], df.iloc[n_train:]


# df_sq: ['id', 'title', 'context', 'question', 'answers']
def get_sq_batch(tkz: PreTrainedTokenizer, df_sq: pd.DataFrame, inds: np.ndarray, inp_len: int, device: torch.device, ques_inp: QuesInp) -> QnaBatch:
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
            q = f'{ctxs[row.context]}. Question: {row.question}'
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
    return QnaBatch(qas=qas, contexts=contexts, toks_seq_len=inp_len, tkz=tkz, device=device, ques_inp=ques_inp)


def qna_loss(logits: torch.Tensor, tokens: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
    tgt_logits = logits[tgt_mask]
    tgt_probs = torch.softmax(tgt_logits, dim=-1)
    tgt_toks = tokens[tgt_mask][..., None]
    tok_probs = torch.gather(tgt_probs, dim=-1, index=tgt_toks)
    tok_logs = torch.log(tok_probs).reshape(len(tgt_mask))
    loss = -(0.99 * torch.mean(tok_logs[:-1]) + 0.01 * tok_logs[-1])
    return loss


BatchIt = Generator[QnaBatch, None, None]

def get_sq_batch_iterator(
        df_sq: pd.DataFrame, tkz: PreTrainedTokenizer, batch_size: int, inp_len: int, device: torch.device, ques_inp: QuesInp,
) -> BatchIt:
    inds = np.arange(len(df_sq))
    batch_off = 0
    while True:
        batch_inds = inds[batch_off:batch_off + batch_size]
        n_cur = len(batch_inds)
        n_rest = batch_size - n_cur
        if n_rest > 0:
            batch_inds = np.concatenate([batch_inds, inds[:n_rest]])
        sq_batch = get_sq_batch(tkz=tkz, df_sq=df_sq, inds=batch_inds, inp_len=inp_len, device=device, ques_inp=ques_inp)
        yield sq_batch
        batch_off += batch_size
        if batch_off >= len(inds):
            batch_off = 0
            np.random.shuffle(inds)


def run_eed_model_on_batch(model: EncoderEmbDecoderModel, batch: QnaBatch) -> torch.Tensor:
    ctxs_toks, other_toks = batch.gen_tensors()
    ctxs_mask = (ctxs_toks > 0).to(batch.device)
    ctx_enc_out: BaseModelOutputWithPastAndCrossAttentions = model.encoder(input_ids=ctxs_toks, attention_mask=ctxs_mask)
    ctx_emb = ctx_enc_out.last_hidden_state[:, 0].unsqueeze(0)

    if batch.ques_inp == QuesInp.Enc:
        q_toks_l, a_toks_l, a_att_masks_l, a_tgt_masks_l = other_toks
        loss = torch.tensor(0, dtype=torch.float32, device=batch.device)
        n_ans = len(a_toks_l)
        for q_toks, a_toks, a_att_mask, a_tgt_mask in zip(q_toks_l, a_toks_l, a_att_masks_l, a_tgt_masks_l):
            q_toks = q_toks.unsqueeze(0)
            q_mask = (q_toks > 0).to(batch.device)
            q_enc_out: BaseModelOutputWithPastAndCrossAttentions = model.encoder(input_ids=q_toks, attention_mask=q_mask)
            q_emb = q_enc_out.last_hidden_state[:, 0].unsqueeze(0)
            ctxq_emb = torch.concatenate([ctx_emb, q_emb], dim=1)
            a_toks = a_toks.repeat(len(a_att_mask), 1)
            a_toks_inp = torch.tril(a_toks)
            a_toks_inp[a_tgt_mask] = batch.tkz.mask_token_id
            a_dec_out: CausalLMOutputWithCrossAttentions = model.decoder(
                input_ids=a_toks_inp, attention_mask=a_att_mask, encoder_hidden_states=ctxq_emb, use_cache=False,
            )
            l = qna_loss(a_dec_out.logits, a_toks, a_tgt_mask)
            loss = loss + l
        loss = loss / n_ans
        return loss

    if batch.ques_inp == QuesInp.Dec:
        qa_toks_l, qa_att_masks_l, qa_tgt_masks_l = other_toks
        loss = torch.tensor(0, dtype=torch.float32, device=batch.device)
        n_qas = len(qa_toks_l)
        for ind in range(n_qas):
            qa_toks, qa_att_mask, qa_tgt_mask = qa_toks_l[ind].unsqueeze(0), qa_att_masks_l[ind], qa_tgt_masks_l[ind]
            qa_toks = qa_toks.repeat(len(qa_att_mask), 1)
            qa_toks_inp = qa_toks * qa_att_mask
            qa_toks_inp[qa_tgt_mask] = batch.tkz.mask_token_id
            dec_out: CausalLMOutputWithCrossAttentions = model.decoder(
                input_ids=qa_toks_inp, attention_mask=qa_att_mask, encoder_hidden_states=ctx_emb, use_cache=False,
            )
            l = qna_loss(dec_out.logits, qa_toks, qa_tgt_mask)
            loss = loss + l
        loss = loss / n_qas
        return loss

    raise Exception(f'Question input type {batch.ques_inp} is not supported.')


