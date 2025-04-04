import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import torch.utils.tensorboard as tb
from torch import nn
from torch.nn.modules import activation
from transformers import PreTrainedTokenizer

from mllm.data.common import DsView, TDs, TBatch
from mllm.utils.utils import gen_dt_str, DT_PAT_RE, parse_dt_str

SUBDIR_PAT_STR = re.compile(r'^[\w-]*?-(%s)-.*$' % DT_PAT_RE)
SUBDIR_PAT = re.compile(SUBDIR_PAT_STR)
DT_PAT = re.compile(DT_PAT_RE)


def get_dt_from_subdir(subdir: str) -> Optional[str]:
    parts = subdir.split('-')
    for part in parts:
        if DT_PAT.match(part):
            return part


def gen_train_subdir(prefix: Optional[str], postfix: Optional[str]) -> str:
    subdir = gen_dt_str()
    if prefix:
        subdir = f'{prefix}-{subdir}'
    if postfix:
        subdir = f'{subdir}-{postfix}'
    return subdir


def find_last_train_subdir(train_root_path: Path, prefix: Optional[str] = None, postfix: Optional[str] = None) -> Optional[Path]:
    dt_last: Optional[datetime] = None
    subdir_last: Optional[str] = None
    for subpath in train_root_path.iterdir():
        if not subpath.is_dir():
            continue
        subdir = subpath.name
        if prefix:
            # print(subdir, subdir.startswith(prefix))
            if not subdir.startswith(prefix):
                continue
            subdir = subdir[len(prefix):]
        if postfix:
            # print(subdir, subdir.endswith(postfix), postfix)
            if not subdir.endswith(postfix):
                continue
            subdir = subdir[:-len(postfix)]
        assert subdir, f'prefix: {prefix}. postfix: {postfix}. subdir: {subpath.name}'
        print(subdir)
        m = SUBDIR_PAT.match(subdir)
        if not m:
            continue
        dt_cur = parse_dt_str(m.group(1))
        if dt_cur is None:
            continue
        if dt_last is None or dt_cur > dt_last:
            dt_last = dt_cur
            subdir_last = subpath.name
    if subdir_last is not None:
        return train_root_path / subdir_last


def find_create_train_path(train_root_path: Path, prefix: Optional[str] = None, postfix: Optional[str] = None, subdir: Optional[str] = None) -> Path:
    if subdir == 'last':
        train_path = find_last_train_subdir(train_root_path, prefix, postfix)
        if train_path is None:
            raise Exception(f'Cannot find last subdirectory of the format `{SUBDIR_PAT_STR}` (prefix = {prefix}, postfix = {postfix}) in {train_root_path}')
    elif subdir:
        train_path = train_root_path / subdir
        assert train_path.exists(), f'Directory {train_path} does not exist'
    else:
        train_subdir = gen_train_subdir(prefix, postfix)
        train_path = train_root_path / train_subdir
        train_path.mkdir(parents=True, exist_ok=True)
    return train_path


def print_grad(model: torch.nn.Module):
    for name, p in model.named_parameters():
        grad = p.grad.cpu().detach().numpy()
        p = p.cpu().detach().numpy()
        eps = 1e-8
        print(name, p.dtype, grad.shape, np.prod(list(grad.shape)), (grad < eps).sum())
        print(' ' * 4, p.min(), p.mean(), p.max())


def calc_print_batches(view_train: DsView[TDs, TBatch], view_val: DsView[TDs, TBatch], batch_size: int, items_name: str) -> tuple[int, int]:
    calc_batches = lambda n_items: n_items // batch_size + (n_items % batch_size > 1)
    n_qs_train, n_qs_val = len(view_train), len(view_val)
    n_batches_train = calc_batches(n_qs_train)
    n_batches_val = calc_batches(n_qs_val)
    print(f'{items_name} train: {n_qs_train}')
    print(f'{items_name} val: {n_qs_val}')
    print(f'Batches train: {n_batches_train}')
    print(f'Batches val: {n_batches_val}')
    return n_batches_train, n_batches_val


def calc_print_batches_multi(views_train: list[DsView[TDs, TBatch]], views_val: list[DsView[TDs, TBatch]], batch_size: int, items_name: str) -> tuple[int, int]:
    calc_batches = lambda n_items: n_items // batch_size + (n_items % batch_size > 1)
    n_qs_train, n_qs_val = sum(len(v) for v in views_train), sum(len(v) for v in views_val)
    n_batches_train = calc_batches(n_qs_train)
    n_batches_val = calc_batches(n_qs_val)
    print(f'{items_name} train: {n_qs_train}')
    print(f'{items_name} val: {n_qs_val}')
    print(f'Batches train: {n_batches_train}')
    print(f'Batches val: {n_batches_val}')
    return n_batches_train, n_batches_val


def concat_tokens(*chunks: torch.Tensor, shuffle: bool = True) ->torch.Tensor:
    if shuffle:
        chunks = list(chunks)
        np.random.shuffle(chunks)
    return torch.concat(chunks, dim=0)


# chunks: input token chunks of the shape [n_docs, n_tokens_per_doc]
def remove_tokens(chunks: torch.Tensor, mask_tok: int, rem_ratio: float = 0.15, rem_conseq_ratio: float = 0.3) -> torch.Tensor:
    res = chunks.clone()
    rv = np.random.rand()
    if rv < 1 / 3:
        p = rem_ratio
        mask = torch.distributions.Bernoulli(probs=p).sample(chunks.size()).to(chunks.device)
        res[mask.bool()] = mask_tok
    elif rv < 2 / 3:
        n = chunks.shape[-1]
        n_rem = int(n * rem_conseq_ratio)
        n_rem = np.random.randint(1, n_rem)
        i = np.random.randint(n - n_rem + 1)
        res[:, i:i + n_rem] = mask_tok
    return res


def calc_params_grads_stats(params: torch.nn.Parameter) -> tuple[tuple[float, float], Optional[tuple[float, float]]]:
    gres = None
    pres = params.mean().detach().cpu().item(), params.std().detach().cpu().item()
    if params.grad is not None:
        gres = params.grad.mean().detach().cpu().item(), params.grad.std().detach().cpu().item()
    return pres, gres


def log_weights_grads_stats(step: int, model: torch.nn.Module, tbsw: tb.SummaryWriter):
    for i, (pname, params) in enumerate(model.named_parameters()):
        pname = f'{i:02d}-{pname}'
        pms, gms = calc_params_grads_stats(params)
        # print(pname, pms, gms)
        weight_mean, weight_std = pms
        tbsw.add_scalar(f'{pname}/WeightMean', weight_mean, step)
        tbsw.add_scalar(f'{pname}/WeightStd', weight_std, step)
        if gms is not None:
            grad_mean, grad_std = gms
            tbsw.add_scalar(f'{pname}/GradMean', grad_mean, step)
            tbsw.add_scalar(f'{pname}/GradStd', grad_std, step)


Activation = Callable[..., nn.Module]


def get_activation_module(act: str) -> Activation:
    # get list from activation submodule as lower-case
    activations_lc = [str(a).lower() for a in activation.__all__]
    if (act := str(act).lower()) in activations_lc:
        # match actual name from lower-case list, return function/factory
        idx = activations_lc.index(act)
        act_name = activation.__all__[idx]
        act_func = getattr(activation, act_name)
        return act_func
    else:
        raise ValueError(f'Cannot find activation function for string <{act}>')


def extend_mask_to_words(mask: np.ndarray, toks_str: list[str]) -> np.ndarray:
    n = len(mask)
    for i in range(1, n):
        if not mask[i] and mask[i - 1] and toks_str[i].startswith('##'):
            mask[i] = True
    for i in range(n - 2, -1, -1):
        if not mask[i] and mask[i + 1] and toks_str[i + 1].startswith('##'):
            mask[i] = True
    return mask


def mask_random_tokens(toks: np.ndarray, tkz: PreTrainedTokenizer, rem_freq: float = 0.33, rem_prob: float = 0.15,
        rem_conseq_freq: float = 0.33, rem_conseq_prob: float = 0.2, rem_conseq_max_len: int = 20,
        rem_conseq_max_times: int = 5) -> np.ndarray:
    res = toks.copy()
    rv = np.random.rand()
    n_total = len(res)
    if rv > rem_freq + rem_conseq_freq:
        return res

    if n_total < 5:
        return res

    if rv <= rem_freq:
        mask: np.ndarray = np.random.rand(n_total) <= rem_prob
    elif rv <= rem_freq + rem_conseq_freq:
        rem_conseq_times = np.random.randint(1, rem_conseq_max_times + 1)
        rem_interval = n_total // rem_conseq_times
        off = 0
        mask = np.full(n_total, False, dtype=bool)
        while off < n_total:
            n_rem = int(n_total * rem_conseq_prob)
            n_rem = np.random.randint(2, max(n_rem, 2) + 1)
            n_rem = min(n_rem, rem_conseq_max_len)
            i = np.random.randint(off, off + rem_interval)
            i1 = max(i - n_rem // 2, 0)
            i2 = min(i1 + n_rem, n_total - 1)
            if i1 < i2:
                mask[i1:i2] = True
            off = max(off + rem_interval, i2 + int(n_rem * 1.5))

    toks_str = [tkz.decode(t) for t in toks]
    mask = extend_mask_to_words(mask, toks_str)
    res[mask] = tkz.mask_token_id
    return res


NEWLINE_PAT = re.compile(r'[\n\r]+', re.M)
STR_DELIM_PAT = re.compile(r'\s+')


def mask_random_words(
        s: str, mask_tok_str: str, rem_freq: float = 0.33, rem_prob: float = 0.15,
        rem_conseq_freq: float = 0.33, rem_conseq_prob: float = 0.2, rem_conseq_max_len: int = 20,
        rem_conseq_max_times: int = 5,
        ) -> Optional[str]:
    rv = np.random.rand()
    if rv < 1 - (rem_freq + rem_conseq_freq):
        return
    lines = NEWLINE_PAT.split(s)
    res = []
    n_total = 0
    for line in lines:
        if not line:
            continue
        words = STR_DELIM_PAT.split(line)
        words = filter(None, words)
        words = list(words)
        if not words:
            continue
        res.append(words)
        n_total += len(words)

    if n_total < 5:
        return

    if rv < 1 - rem_conseq_freq:
        mask = np.random.rand(n_total) <= rem_prob
    else:
        rem_conseq_times = np.random.randint(1, rem_conseq_max_times + 1)
        rem_interval = n_total // rem_conseq_times
        off = 0
        mask = np.full(n_total, False, dtype=bool)
        while off < n_total:
            n_rem = int(n_total * rem_conseq_prob)
            n_rem = np.random.randint(2, max(n_rem, 2) + 1)
            n_rem = min(n_rem, rem_conseq_max_len)
            i = np.random.randint(off, off + rem_interval)
            i1 = max(i - n_rem // 2, 0)
            i2 = min(i1 + n_rem, n_total - 1)
            if i1 < i2:
                mask[i1:i2] = True
            off = max(off + rem_interval, i2 + int(n_rem * 1.5))

    im = 0
    for words in res:
        for iw in range(len(words)):
            if mask[im]:
                words[iw] = mask_tok_str
            im += 1

    return '\n'.join([' '.join(words) for words in res])


