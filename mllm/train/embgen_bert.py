from enum import Enum
from typing import Optional, Generator

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
    qa_toks: list[np.ndarray]
    q_toks: list[np.ndarray]
    a_toks: list[np.ndarray]
    qa_att_masks: list[np.ndarray]
    qa_tgt_masks: list[np.ndarray]
    ctx_toks: np.ndarray
    device: Optional[torch.device] = None
    qa_toks_t: list[torch.Tensor] = []
    q_toks_t: list[torch.Tensor] = []
    a_toks_t: list[torch.Tensor] = []
    qa_att_mask_t: list[torch.Tensor] = []
    qa_tgt_mask_t: list[torch.Tensor] = []
    ctx_toks_t: Optional[torch.Tensor] = None

    def __init__(
            self, qas: list[tuple[str, str]], contexts: list[str], toks_seq_len: int, tkz: PreTrainedTokenizer,
            device: Optional[torch.device] = None,
    ):
        self.qas = qas
        self.contexts = contexts
        self.toks_seq_len = toks_seq_len
        self.tkz = tkz
        self.device = device
        self._process()

    def _process(self):
        # # Question + Answer
        # q_toks_l, a_toks_l, a_att_masks_l, a_tgt_masks_l = [], [], [], []
        # for q, a in self.qas:
        #     q_toks: list[int] = self.tkz(q).input_ids
        #     a_toks: list[int] = self.tkz(a).input_ids
        #     assert q_toks[0] == a_toks[0] == self.tkz.cls_token_id, f'q_toks[0] = {q_toks[0]}. a_toks[0] = {a_toks[0]}'
        #     assert q_toks[-1] == a_toks[-1] == self.tkz.sep_token_id, f'q_toks[-1] = {q_toks[-1]}. a_toks[-1] = {a_toks[-1]}'
        #     q_toks_l.append(np.array(q_toks, dtype=int))
        #     a_toks_l.append(np.array(a_toks, dtype=int))
        #
        #     n_q_toks, n_a_toks = len(q_toks), len(a_toks)
        #     q_mask = np.ones((n_a_toks, n_q_toks + 1), dtype=int)
        #     a_att_mask = np.ones((n_a_toks, n_a_toks), dtype=int)
        #     a_att_mask = np.tril(a_att_mask, k=-1)
        #     a_tgt_mask = np.eye(n_a_toks, dtype=int)
        #     qa_att_mask = np.concatenate([q_mask, a_att_mask], axis=1)
        #     qa_tgt_mask = np.concatenate([q_mask * 0, a_tgt_mask], axis=1).astype(bool)
        #     qa_att_masks_l.append(qa_att_mask)
        #     qa_tgt_masks_l.append(qa_tgt_mask)
        # self.qa_toks = qa_toks_l
        # self.qa_att_masks = qa_att_masks_l
        # self.qa_tgt_masks = qa_tgt_masks_l

        # Question + Answer
        qa_toks_l, qa_att_masks_l, qa_tgt_masks_l = [], [], []
        for q, a in self.qas:
            q_toks: list[int] = self.tkz(q).input_ids
            a_toks: list[int] = self.tkz(a).input_ids
            assert q_toks[0] == a_toks[0] == self.tkz.cls_token_id, f'q_toks[0] = {q_toks[0]}. a_toks[0] = {a_toks[0]}'
            assert q_toks[-1] == a_toks[-1] == self.tkz.sep_token_id, f'q_toks[-1] = {q_toks[-1]}. a_toks[-1] = {a_toks[-1]}'
            q_toks, a_toks = q_toks[0:-1], a_toks[1:]
            qa_toks = [*q_toks, self.tkz.sep_token_id, *a_toks]
            # TODO: More robust rule
            if len(qa_toks) > 40:
                continue
            qa_toks_l.append(np.array(qa_toks, dtype=int))

            n_q_toks, n_a_toks = len(q_toks), len(a_toks)
            q_mask = np.ones((n_a_toks, n_q_toks + 1), dtype=int)
            a_att_mask = np.ones((n_a_toks, n_a_toks), dtype=int)
            a_att_mask = np.tril(a_att_mask, k=-1)
            a_tgt_mask = np.eye(n_a_toks, dtype=int)
            qa_att_mask = np.concatenate([q_mask, a_att_mask], axis=1)
            qa_tgt_mask = np.concatenate([q_mask * 0, a_tgt_mask], axis=1).astype(bool)
            qa_att_masks_l.append(qa_att_mask)
            qa_tgt_masks_l.append(qa_tgt_mask)
        self.qa_toks = qa_toks_l
        self.qa_att_masks = qa_att_masks_l
        self.qa_tgt_masks = qa_tgt_masks_l

        # Contexts
        ctxs = []
        max_ctx_chunks = 3
        for ctx in self.contexts:
            ctx_toks = self.tkz(ctx).input_ids
            n_pad = self.toks_seq_len - len(ctx_toks) % self.toks_seq_len
            assert self.tkz.pad_token_id is not None
            ctx_toks = np.pad(ctx_toks, (0, n_pad), constant_values=self.tkz.pad_token_id)
            # ctx_toks = ctx_toks.reshape((-1, self.toks_seq_len))
            ctx_toks = ctx_toks[:self.toks_seq_len][None]
            ctxs.append(ctx_toks[:max_ctx_chunks])
        ctxs_all = np.concatenate(ctxs)
        self.ctx_toks = ctxs_all

        # ctxs_lens = [len(c) for c in ctxs]
        # qas_lens = [len(qa) for qa in self.qa_toks]
        # print(f'Contexts: {ctxs_lens}. {ctxs_all.shape}')
        # print(f'QAs: {qas_lens}.')

    def _to_tensor_single(self, arr: np.ndarray) -> torch.Tensor:
        res = torch.from_numpy(arr)
        if self.device is not None:
            res = res.to(self.device)
        return res

    def _to_tensor_multi(self, arr: list[np.ndarray]) -> list[torch.Tensor]:
        return [self._to_tensor_single(x) for x in arr]

    def gen_tensors(self) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        if self.ctx_toks_t is None:
            self.qa_toks_t = self._to_tensor_multi(self.qa_toks)
            self.qa_att_mask_t = self._to_tensor_multi(self.qa_att_masks)
            self.qa_tgt_mask_t = self._to_tensor_multi(self.qa_tgt_masks)
            self.ctx_toks_t = self._to_tensor_single(self.ctx_toks)
        return self.qa_toks_t, self.qa_att_mask_t, self.qa_tgt_mask_t, self.ctx_toks_t


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
def get_sq_batch(tkz: PreTrainedTokenizer, df_sq: pd.DataFrame, inds: np.ndarray, inp_len: int, device: torch.device) -> QnaBatch:
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
            q = f'{ctxs[row.context]}. {row.question}'
            qa = q, answer
            qas.add(qa)
    contexts = [f'{val}. {key}' for key, val in ctxs.items()]
    qas = list(qas)
    n_qas, n_batch = len(qas), len(df_b)
    if n_qas > n_batch:
        np.random.shuffle(qas)
        qas = qas[:n_batch]
    return QnaBatch(qas=qas, contexts=contexts, toks_seq_len=inp_len, tkz=tkz, device=device)


def qna_loss(logits: torch.Tensor, tokens: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
    tgt_logits = logits.masked_select(tgt_mask.unsqueeze(-1))
    tgt_logits = tgt_logits.reshape(logits.shape[0], logits.shape[2])
    tgt_probs = torch.softmax(tgt_logits, dim=-1)
    tgt_toks = tokens.masked_select(tgt_mask).unsqueeze(-1)
    tok_probs = torch.gather(tgt_probs, dim=-1, index=tgt_toks)
    tok_logs = torch.log(tok_probs).reshape(len(tgt_mask))
    loss = -(0.95 * torch.mean(tok_logs[:-1]) + 0.05 * tok_logs[-1])
    return loss


BatchIt = Generator[QnaBatch, None, None]

def get_sq_batch_iterator(df_sq: pd.DataFrame, tkz: PreTrainedTokenizer, batch_size: int, inp_len: int, device: torch.device) -> BatchIt:
    inds = np.arange(len(df_sq))
    batch_off = 0
    while True:
        batch_inds = inds[batch_off:batch_off + batch_size]
        n_cur = len(batch_inds)
        n_rest = batch_size - n_cur
        if n_rest > 0:
            batch_inds = np.concatenate([batch_inds, inds[:n_rest]])
        sq_batch = get_sq_batch(tkz=tkz, df_sq=df_sq, inds=batch_inds, inp_len=inp_len, device=device)
        yield sq_batch
        batch_off += batch_size
        if batch_off >= len(inds):
            batch_off = 0
            np.random.shuffle(inds)


def run_eed_model_on_batch(model: EncoderEmbDecoderModel, batch: QnaBatch) -> torch.Tensor:
    qas_toks, qa_att_masks, qa_tgt_masks, ctxs_toks = batch.gen_tensors()
    ctxs_mask = (ctxs_toks > 0).to(batch.device)
    enc_out: BaseModelOutputWithPastAndCrossAttentions = model.encoder(input_ids=ctxs_toks, attention_mask=ctxs_mask)
    enc_emb = enc_out.last_hidden_state[:, 0].unsqueeze(0)
    loss = torch.tensor(0, dtype=torch.float32, device=batch.device)
    n_qas = len(qas_toks)
    for ind in range(n_qas):
        qa_toks, qa_att_mask, qa_tgt_mask = qas_toks[ind].unsqueeze(0), qa_att_masks[ind], qa_tgt_masks[ind]
        qa_toks = qa_toks.repeat(len(qa_att_mask), 1)
        qa_toks_inp = qa_toks * qa_att_mask
        dec_out: CausalLMOutputWithCrossAttentions = model.decoder(
            input_ids=qa_toks_inp, attention_mask=qa_att_mask, encoder_hidden_states=enc_emb
        )
        l = qna_loss(dec_out.logits, qa_toks, qa_tgt_mask)
        loss = loss + l
    loss = loss / n_qas
    return loss



