from typing import Union

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
        loss_tgt = torch.tensor(0, dtype=torch.float32, device=cos_pred.device)
        loss_nontgt = torch.tensor(0, dtype=torch.float32, device=cos_pred.device)
        for i in range(n_docs):
            probs_tgt = 1 - torch.masked_select(cos_pred[i], mask_gt[i])
            probs_nontgt = torch.masked_select(cos_pred[i], ~mask_gt[i])
            probs_nontgt = torch.maximum(probs_nontgt - self.margin, self.zero)
            lt, lnt = torch.mean(probs_tgt), torch.mean(probs_nontgt)
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


