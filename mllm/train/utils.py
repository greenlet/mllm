import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from mllm.data.common import DsView, TDs, TBatch
from mllm.utils.utils import gen_dt_str, DT_PAT_RE, parse_dt_str

SUBDIR_PAT_STR = re.compile(r'^\w+\-(%s)-.+$' % DT_PAT_RE)
SUBDIR_PAT = re.compile(SUBDIR_PAT_STR)
DT_PAT = re.compile(r'\d{8}_\d{6}')


def get_dt_from_subdir(subdir: str) -> Optional[str]:
    parts = subdir.split('-')
    for part in parts:
        if DT_PAT.match(part):
            return part


def gen_train_subdir(prefix: str, postfix: Optional[str]) -> str:
    dt_str = gen_dt_str()
    subdir = f'{prefix}-{dt_str}'
    if postfix:
        subdir = f'{subdir}-{postfix}'
    return subdir


def find_last_train_subdir(train_root_path: Path) -> Optional[Path]:
    dt_last: Optional[datetime] = None
    subdir_last: Optional[str] = None
    for subpath in train_root_path.iterdir():
        if not subpath.is_dir():
            continue
        m = SUBDIR_PAT.match(subpath.name)
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
        train_path = find_last_train_subdir(train_root_path)
        if train_path is None:
            raise Exception(f'Cannot find last subdirectory of the format `{SUBDIR_PAT_STR}` in {train_root_path}')
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

