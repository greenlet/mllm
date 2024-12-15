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

