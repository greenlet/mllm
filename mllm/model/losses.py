from enum import Enum
from typing import Union

from sympy import N
import torch
from torch import nn



class RankProbLoss(nn.Module):
    def __init__(self, target_weight: float = 0.5):
        super().__init__()
        self.target_weight = target_weight
        self.register_buffer('prob_cap', torch.scalar_tensor(1e-6))

    def forward(self, prob_pred: list[torch.Tensor], mask_gt: Union[torch.Tensor, list[torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_batch = len(prob_pred)
        losses_tgt = torch.zeros(n_batch, dtype=torch.float32, device=prob_pred[0].device)
        losses_nontgt = torch.zeros(n_batch, dtype=torch.float32, device=prob_pred[0].device)
        for i in range(n_batch):
            prob_tgt = torch.masked_select(prob_pred[i], mask_gt[i])
            prob_nontgt = 1 - torch.masked_select(prob_pred[i], ~mask_gt[i])

            prob_tgt = torch.maximum(prob_tgt, self.prob_cap)
            prob_nontgt = torch.maximum(prob_nontgt, self.prob_cap)
            losses_tgt[i] = -torch.mean(torch.log(prob_tgt))
            losses_nontgt[i] = -torch.mean(torch.log(prob_nontgt))
            # losses_tgt[i] = -torch.min(torch.log(prob_tgt))
            # losses_nontgt[i] = -torch.min(torch.log(prob_nontgt))

        loss_tgt = torch.mean(losses_tgt)
        loss_nontgt = torch.mean(losses_nontgt)
        # loss = self.target_weight * loss_tgt + (1 - self.target_weight) * loss_nontgt
        loss = loss_tgt + loss_nontgt
        return loss, loss_tgt, loss_nontgt

    def forward_1(self, prob_pred: torch.Tensor, mask_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prob_pred = prob_pred.squeeze()
        prob_tgt = torch.masked_select(prob_pred, mask_gt)
        prob_nontgt = 1 - torch.masked_select(prob_pred, ~mask_gt)
        prob_tgt = torch.maximum(prob_tgt, self.prob_cap)
        prob_nontgt = torch.maximum(prob_nontgt, self.prob_cap)
        loss_tgt = -torch.mean(torch.log(prob_tgt))
        loss_nontgt = -torch.mean(torch.log(prob_nontgt))
        loss = self.target_weight * loss_tgt + (1 - self.target_weight) * loss_nontgt
        return loss, loss_tgt, loss_nontgt


def ranker_prob_loss_softmax(prob_pred: list[torch.Tensor], mask_gt: Union[torch.Tensor, list[torch.Tensor]]) -> torch.Tensor:
    n_batch = len(prob_pred)
    losses = torch.zeros(n_batch, dtype=torch.float32, device=prob_pred[0].device)
    for i in range(n_batch):
        prob = torch.masked_select(prob_pred[i], mask_gt[i])
        losses[i] = -torch.sum(torch.log(prob))
    loss = torch.mean(losses)
    return loss


def encdec_prob_loss_softmax(logits_pred: torch.Tensor, tokens_gt: torch.Tensor) -> torch.Tensor:
    tokens_gt = tokens_gt.to(torch.int64).unsqueeze(-1)
    probs_pred = torch.softmax(logits_pred, dim=-1)
    probs_gt = torch.gather(probs_pred, dim=2, index=tokens_gt)
    loss = -torch.mean(torch.log(probs_gt))
    return loss


def encdec_prob_loss_sigmoid(logits_pred: torch.Tensor, tokens_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = logits_pred.device
    tokens_gt = tokens_gt.to(torch.int64).unsqueeze(-1)
    probs_pred = torch.sigmoid(logits_pred)
    prob_cap = torch.tensor(1e-6, dtype=torch.float32, device=device)
    probs_gt = torch.gather(probs_pred, dim=2, index=tokens_gt)
    probs_gt = torch.maximum(probs_gt, prob_cap)
    loss_gt = -torch.mean(torch.log(probs_gt))
    loss_nongt = torch.tensor(0, dtype=torch.float32, device=device)
    for i in range(probs_pred.shape[0]):
        mask = torch.full((logits_pred.shape[-2], logits_pred.shape[-1],), True, device=device)
        mask = mask.scatter(1, tokens_gt[i], 0)
        probs_nongt = 1 - probs_pred[i][mask]
        probs_nongt = torch.maximum(probs_nongt, prob_cap)
        loss_nongt += -torch.mean(torch.log(probs_nongt))
    loss_nongt = loss_nongt / logits_pred.shape[0]
    loss = loss_gt + loss_nongt
    return loss_gt, loss_nongt, loss


class EncdecProbLossSigmoid(nn.Module):
    def __init__(self, seq_len: int, n_tokens: int, device: torch.device):
        super().__init__()
        mask = torch.full((seq_len, n_tokens), True, device=device)
        self.register_buffer('mask', mask)

    def forward(self, logits_pred: torch.Tensor, tokens_gt: torch.Tensor) -> torch.Tensor:
        tokens_gt = tokens_gt.to(torch.int64).unsqueeze(-1)
        probs_pred = torch.sigmoid(logits_pred)
        probs_gt = torch.gather(probs_pred, dim=2, index=tokens_gt)
        loss_gt = -torch.mean(torch.log(probs_gt))
        loss_nongt = 0
        for i in range(probs_pred.shape[0]):
            self.mask.scatter_(1, tokens_gt[i], 0)
            loss_nongt += -torch.mean(torch.log(1 - probs_pred[i][self.mask]))
            self.mask.scatter_(1, tokens_gt[i], 1)
        loss_nongt = loss_nongt / logits_pred.shape[0]
        loss = loss_gt + loss_nongt
        return loss


# prob_pred: (n_docs, n_qs)
# mask_gt: (n_docs, n_qs)
def ranker_cos_loss(cos_pred: torch.Tensor, mask_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_gt = mask_gt.to(torch.bool)
    n_docs = len(cos_pred)
    loss_tgt = torch.tensor(0, dtype=torch.float32, device=cos_pred.device)
    loss_nontgt = torch.tensor(0, dtype=torch.float32, device=cos_pred.device)
    prob_cap = torch.tensor(1e-6, dtype=torch.float32, device=cos_pred.device)
    for i in range(n_docs):
        probs_tgt = torch.masked_select(cos_pred[i], mask_gt[i])
        probs_nontgt = -torch.masked_select(cos_pred[i], ~mask_gt[i])
        probs_tgt = (probs_tgt + 1) / 2
        probs_nontgt = (probs_nontgt + 1) / 2
        probs_tgt = torch.maximum(probs_tgt, prob_cap)
        probs_nontgt = torch.maximum(probs_nontgt, prob_cap)
        lt, lnt = -torch.mean(torch.log(probs_tgt)), -torch.mean(torch.log(probs_nontgt))
        loss_tgt = loss_tgt + lt
        loss_nontgt = loss_nontgt + lnt
    loss_tgt = loss_tgt / n_docs
    loss_nontgt = loss_nontgt / n_docs
    loss = (loss_tgt + loss_nontgt) / 2
    if torch.isnan(loss).any():
        print('!!!', torch.isnan(cos_pred).any())
        print(mask_gt)
        import sys
        sys.exit(0)
    return loss, loss_tgt, loss_nontgt


class RankerCosEmbLoss(nn.Module):
    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.register_buffer('margin', torch.scalar_tensor(margin))
        self.register_buffer('zero', torch.scalar_tensor(0.0))

    # prob_pred: (n_docs, n_qs)
    # mask_gt: (n_docs, n_qs)
    def forward(self, cos_pred: torch.Tensor, mask_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_gt = mask_gt.to(torch.bool)
        n_docs = len(cos_pred)
        # loss_tgt = torch.tensor(0, dtype=torch.float32, device=cos_pred.device)
        # loss_nontgt = torch.tensor(0, dtype=torch.float32, device=cos_pred.device)
        losses_tgt = torch.zeros_like(cos_pred[:, 0], device=cos_pred.device)
        losses_nontgt = torch.zeros_like(cos_pred[:, 0], device=cos_pred.device)
        for i in range(n_docs):
            probs_tgt = 1 - torch.masked_select(cos_pred[i], mask_gt[i])
            probs_nontgt = torch.masked_select(cos_pred[i], ~mask_gt[i])
            probs_nontgt = torch.maximum(probs_nontgt - self.margin, self.zero)
            # lt, lnt = torch.mean(probs_tgt), torch.mean(probs_nontgt)
            # loss_tgt = loss_tgt + lt
            # loss_nontgt = loss_nontgt + lnt
            lt, lnt = torch.max(probs_tgt), torch.max(probs_nontgt)
            losses_tgt[i], losses_nontgt[i] = lt, lnt
        # loss_tgt = loss_tgt / n_docs
        # loss_nontgt = loss_nontgt / n_docs
        loss_tgt, loss_nontgt = torch.max(losses_tgt), torch.max(losses_nontgt)
        loss = (loss_tgt + loss_nontgt) / 2
        if torch.isnan(loss).any():
            print('!!!', torch.isnan(cos_pred).any())
            print(mask_gt)
            import sys
            sys.exit(0)
        return loss, loss_tgt, loss_nontgt


class EncdecMaskPadBatchLoss(nn.Module):
    msk_tok_ind: int
    pad_tok_ind: int
    reg_weight: float
    msk_weight: float
    pad_weight: float
    total_weight: float
    # prob_cap: float

    def __init__(
            self, msk_tok_ind: int, pad_tok_ind: int, reg_weight: float = 1, msk_weight: float = 1, pad_weight: float = 0.01,
            prob_cap: float = 1e-6):
        super().__init__()
        assert 0 <= prob_cap <= 1, f'prob_cap (={prob_cap}) must pertain to [0, 1] interval'
        assert reg_weight > 0, f'reg_weight (={reg_weight}) must be > 0'
        assert msk_weight > 0, f'msk_weight (={msk_weight}) must be > 0'
        assert pad_weight > 0, f'pad_weight (={pad_weight}) must be > 0'
        self.msk_tok_ind = msk_tok_ind
        self.pad_tok_ind = pad_tok_ind
        self.reg_weight = reg_weight
        self.msk_weight = msk_weight
        self.pad_weight = pad_weight
        self.total_weight = self.reg_weight + self.msk_weight + self.pad_weight
        self.nonpad_weight = 1 - pad_weight
        self.register_buffer('prob_cap', torch.scalar_tensor(prob_cap))

    # logits_pred: (batch_size, inp_len, vocab_size)
    # tokens_gt: (batch_size, inp_len)
    def forward(self, logits_pred: torch.Tensor, tokens_gt: torch.Tensor) -> torch.Tensor:
        # tokens_gt: (batch_size, inp_len, 1)
        tokens_gt = tokens_gt.to(torch.int64).unsqueeze(-1)
        # mask_msk: (batch_size, inp_len, 1)
        mask_msk = tokens_gt == self.msk_tok_ind
        # mask_pad: (batch_size, inp_len, 1)
        mask_pad = tokens_gt == self.pad_tok_ind
        # mask_reg: (batch_size, inp_len, 1)
        mask_reg = ~mask_msk & ~mask_pad

        # probs_pred: (batch_size, inp_len, vocab_size)
        probs_pred = torch.softmax(logits_pred, dim=-1)
        # probs_gt: (batch_size, inp_len, 1)
        probs_gt = torch.gather(probs_pred, dim=2, index=tokens_gt)
        # probs_gt = torch.maximum(probs_gt, self.prob_cap)

        # probs_gt_reg: (n_reg_tokens, )
        # probs_gt_msk: (n_msk_tokens, )
        # probs_gt_pad: (n_pad_tokens, )
        # n_reg_tokens + n_msk_tokens + n_pad_tokens = batch_size * inp_len
        probs_gt_reg, probs_gt_msk, probs_gt_pad = probs_gt[mask_reg], probs_gt[mask_msk], probs_gt[mask_pad]

        # loss_reg: (1,)
        # loss_msk: (1,)
        # loss_pad: (1,)
        loss_reg = torch.zeros((1,), dtype=torch.float32, device=probs_gt.device)
        loss_msk, loss_pad = loss_reg, loss_reg
        total_weight = 0
        if probs_gt_reg.size()[0] > 0:
            loss_reg = -torch.mean(torch.log(probs_gt_reg))
            total_weight += self.reg_weight
        if probs_gt_msk.size()[0] > 0:
            loss_msk = -torch.mean(torch.log(probs_gt_msk))
            total_weight += self.msk_weight
        if probs_gt_pad.size()[0] > 0:
            loss_pad = -torch.mean(torch.log(probs_gt_pad))
            total_weight += self.pad_weight
        # loss: (1,)
        loss = (loss_reg * self.reg_weight + loss_msk * self.msk_weight + loss_pad * self.pad_weight) / total_weight
        return loss


class EncdecMaskPadItemLoss(nn.Module):
    msk_tok_id: int
    spc_tok_ids: list[int]
    reg_weight: float
    msk_weight: float
    spc_weight: float
    # prob_cap: float

    def __init__(
            self, msk_tok_id: int, spc_tok_ids: list[int], reg_weight: float = 1, msk_weight: float = 1, spc_weight: float = 0.01,
            prob_cap: float = 1e-6):
        super().__init__()
        assert 0 <= prob_cap <= 1, f'prob_cap (={prob_cap}) must pertain to [0, 1] interval'
        assert reg_weight > 0, f'reg_weight (={reg_weight}) must be > 0'
        assert msk_weight > 0, f'msk_weight (={msk_weight}) must be > 0'
        assert spc_weight > 0, f'spc_weight (={spc_weight}) must be > 0'
        self.msk_tok_id = msk_tok_id
        self.spc_tok_ids = spc_tok_ids
        self.reg_weight = reg_weight
        self.msk_weight = msk_weight
        self.spc_weight = spc_weight
        self.register_buffer('prob_cap', torch.scalar_tensor(prob_cap))

    # logits_pred: (batch_size, inp_len, vocab_size)
    # tokens_inp: (batch_size, inp_len)
    # tokens_tgt: (batch_size, inp_len)
    def forward(self, logits_pred: torch.Tensor, tokens_inp: torch.Tensor, tokens_tgt: torch.Tensor, **kwargs) -> dict[str, torch.Tensor] | torch.Tensor:
        # (batch_size, inp_len, 1)
        toks_inp = tokens_inp.to(torch.int64).unsqueeze(-1)
        toks_tgt = tokens_tgt.to(torch.int64).unsqueeze(-1)

        # (batch_size, inp_len, 1)
        mask_msk = toks_inp == self.msk_tok_id

        # (batch_size, inp_len, 1)
        mask_spc = None
        for spc_tok_id in self.spc_tok_ids:
            m_spc = toks_inp == spc_tok_id
            if mask_spc is None:
                mask_spc = m_spc
            else:
                mask_spc = mask_spc | m_spc
        assert mask_spc is not None

        # (batch_size, inp_len, 1)
        mask_reg = ~mask_msk & ~mask_spc

        # probs_pred: (batch_size, inp_len, vocab_size)
        probs_pred = torch.softmax(logits_pred, dim=-1)
        # probs_gt: (batch_size, inp_len, 1)
        probs_gt = torch.gather(probs_pred, dim=2, index=toks_tgt)
        # probs_gt = torch.maximum(probs_gt, self.prob_cap)

        n_batch = logits_pred.shape[0]
        loss = torch.zeros((1,), dtype=torch.float32, device=probs_gt.device)
        loss_reg, loss_msk, loss_spc = loss, loss, loss
        n_reg, n_msk, n_spc = 0, 0, 0
        for ib in range(n_batch):
            probs_gt_i, mask_reg_i, mask_msk_i, mask_spc_i = probs_gt[ib], mask_reg[ib], mask_msk[ib], mask_spc[ib]
            # probs_gt_reg: (n_reg_toks, )
            probs_gt_reg = probs_gt_i[mask_reg_i]
            # probs_gt_msk: (n_msk_toks, )
            probs_gt_msk = probs_gt_i[mask_msk_i]
            # probs_gt_spc: (n_spc_toks, )
            probs_gt_spc = probs_gt_i[mask_spc_i]
            # n_reg_toks + n_msk_toks + n_spc_toks = inp_len

            # loss_reg: (1,)
            # loss_msk: (1,)
            # loss_spc: (1,)
            loss_i = torch.zeros((1,), dtype=torch.float32, device=probs_gt.device)
            total_weight = 0
            if probs_gt_reg.size()[0] > 0:
                loss_reg_i = -torch.mean(torch.log(probs_gt_reg))
                loss_i = loss_i + loss_reg_i * self.reg_weight
                total_weight += self.reg_weight
                loss_reg = loss_reg + loss_reg_i
                n_reg += 1
            if probs_gt_msk.size()[0] > 0:
                loss_msk_i = -torch.mean(torch.log(probs_gt_msk))
                loss_i = loss_i + loss_msk_i * self.msk_weight
                total_weight += self.msk_weight
                loss_msk = loss_msk + loss_msk_i
                n_msk += 1
            if probs_gt_spc.size()[0] > 0:
                loss_spc_i = -torch.mean(torch.log(probs_gt_spc))
                loss_i = loss_i + loss_spc_i * self.spc_weight
                total_weight += self.spc_weight
                loss_spc = loss_spc + loss_spc_i
                n_spc += 1
            loss_i = loss_i / total_weight
            loss = loss + loss_i
        # loss: (1,)
        loss = loss / n_batch
        res = {'loss': loss}
        if n_reg > 0:
            res['reg_toks_loss'] = loss_reg / n_reg
        if n_msk > 0:
            res['msk_toks_loss'] = loss_msk / n_msk
        if n_spc > 0:
            res['spc_toks_loss'] = loss_spc / n_spc
        return res


class EncdecPadBatchLoss(nn.Module):
    pad_tok_id: int
    weight_reg: float
    weight_pad: float

    def __init__(self, pad_tok_id: int, weight_reg: float = 1, weight_pad: float = 0.1):
        super().__init__()
        self.pad_tok_id = pad_tok_id
        self.weight_reg = weight_reg
        self.weight_pad = weight_pad

    # logits_pred: [batch_size, inp_len, vocab_size]
    # tokens_tgt: [batch_size, tgt_len]
    def forward(self, logits_pred: torch.Tensor, tokens_tgt: torch.Tensor) -> torch.Tensor:
        # [batch_size, tgt_len, 1]
        toks_tgt = tokens_tgt.to(torch.int64).unsqueeze(-1)

        # [batch_size, tgt_len, vocab_size]
        logits_pred = logits_pred[:, :toks_tgt.shape[1]]
        # [batch_size, tgt_len, vocab_size]
        probs_pred = torch.softmax(logits_pred, dim=-1)
        # [batch_size, tgt_len, 1]
        probs_tgt = torch.gather(probs_pred, dim=2, index=toks_tgt)

        # [batch_size, tgt_len]
        mask_pad = tokens_tgt == self.pad_tok_id
        # [batch_size, tgt_len]
        mask_reg = ~mask_pad

        probs_reg = probs_tgt[mask_reg]
        loss_reg = -torch.mean(torch.log(probs_reg))
        probs_pad = probs_tgt[mask_pad]
        loss_pad = -torch.mean(torch.log(probs_pad))
        loss = self.weight_reg * loss_reg + self.weight_pad * loss_pad
        loss = loss / (self.weight_reg + self.weight_pad)
        return loss


class RankLossType(str, Enum):
    Avg = 'avg'
    Max = 'max'
    Lifted = 'lft'


class RankerEmbLoss(nn.Module):
    rank_type: RankLossType

    def __init__(self, rank_type: RankLossType, margin: float = 0.0):
        super().__init__()
        self.rank_type = rank_type
        self.register_buffer('margin', torch.scalar_tensor(margin))
        self.register_buffer('zero', torch.scalar_tensor(0.0))

    # prob_pred: (n_docs, n_qs)
    # mask_gt: (n_docs, n_qs)
    def forward(self, cos_pred: torch.Tensor, mask_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_gt = mask_gt.to(torch.bool)
        n_docs, n_qs = cos_pred.shape

        if self.rank_type == RankLossType.Avg:
            loss_tgt = torch.tensor(0, dtype=torch.float32, device=cos_pred.device)
            loss_nontgt = torch.tensor(0, dtype=torch.float32, device=cos_pred.device)
            for i in range(n_docs):
                probs_tgt = 1 - torch.masked_select(cos_pred[i], mask_gt[i])
                probs_nontgt = torch.masked_select(cos_pred[i], ~mask_gt[i])
                probs_nontgt = torch.maximum(probs_nontgt - self.margin, self.zero)
                lt, lnt = torch.mean(probs_tgt), torch.mean(probs_nontgt)
                loss_tgt, loss_nontgt = loss_tgt + lt, loss_nontgt + lnt
            loss_tgt = loss_tgt / n_docs
            loss_nontgt = loss_nontgt / n_docs
            loss = (loss_tgt + loss_nontgt) / 2
        elif self.rank_type == RankLossType.Max:
            losses_tgt = torch.zeros_like(cos_pred[:, 0], device=cos_pred.device)
            losses_nontgt = torch.zeros_like(cos_pred[:, 0], device=cos_pred.device)
            for i in range(n_docs):
                probs_tgt = 1 - torch.masked_select(cos_pred[i], mask_gt[i])
                probs_nontgt = torch.masked_select(cos_pred[i], ~mask_gt[i])
                probs_nontgt = torch.maximum(probs_nontgt - self.margin, self.zero)
                lt, lnt = torch.max(probs_tgt), torch.max(probs_nontgt)
                losses_tgt[i], losses_nontgt[i] = lt, lnt
            loss_tgt, loss_nontgt = torch.max(losses_tgt), torch.max(losses_nontgt)
            loss = (loss_tgt + loss_nontgt) / 2
        elif self.rank_type == RankLossType.Lifted:
            loss = torch.tensor(0, dtype=torch.float32, device=cos_pred.device)
            loss_tgt = torch.tensor(0, dtype=torch.float32, device=cos_pred.device)
            loss_nontgt = torch.tensor(0, dtype=torch.float32, device=cos_pred.device)
            n_pos = 0
            for i in range(n_docs):
                for j in range(n_qs):
                    if not mask_gt[i, j]: continue
                    probs_tgt = 1 - cos_pred[i, j]
                    probs_nontgt_row = torch.masked_select(cos_pred[i], ~mask_gt[i])
                    probs_nontgt_col = torch.masked_select(cos_pred[:, j], ~mask_gt[:, j])
                    probs_nontgt_row = torch.exp(probs_nontgt_row - self.margin)
                    probs_nontgt_col = torch.exp(probs_nontgt_col - self.margin)
                    probs_nontgt = torch.log(probs_nontgt_row.sum() + probs_nontgt_col.sum())
                    lt = torch.maximum(probs_tgt + probs_nontgt, self.zero)
                    lt = torch.square(lt)
                    loss = loss + lt
                    loss_tgt = loss_tgt + probs_tgt
                    loss_nontgt = loss_nontgt + probs_nontgt
                    n_pos += 1
            dnm = 2 * n_pos
            loss, loss_tgt, loss_nontgt = loss / dnm, loss_tgt / dnm, loss_nontgt / dnm
        else:
            raise Exception(f'Rank loss type {self.rank_type} is not supported')

        if torch.isnan(loss).any():
            print('!!!', torch.isnan(cos_pred).any())
            print(mask_gt)
            import sys
            sys.exit(0)
        return loss, loss_tgt, loss_nontgt



