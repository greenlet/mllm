import numpy as np
import torch


def print_grad(model: torch.nn.Module):
    for name, p in model.named_parameters():
        grad = p.grad.cpu().detach().numpy()
        p = p.cpu().detach().numpy()
        eps = 1e-8
        print(name, p.dtype, grad.shape, np.prod(list(grad.shape)), (grad < eps).sum())
        print(' ' * 4, p.min(), p.mean(), p.max())

