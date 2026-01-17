import csv
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional, Union, TypeVar

import pandas as pd
from torch import optim


DT_PAT = '%Y%m%d_%H%M%S'
DT_PAT_RE = r'\d{8}_\d{6}'


def gen_dt_str(dt: Optional[datetime] = None) -> str:
    dt = dt if dt is not None else datetime.now()
    return dt.strftime(DT_PAT)


def parse_dt_str(dt_str: str, silent: bool = True) -> Optional[datetime]:
    try:
        return datetime.strptime(dt_str, DT_PAT)
    except Exception:
        pass


def write_tsv(df: pd.DataFrame, fpath: Path, **kwargs):
    df.to_csv(fpath, sep='\t', header=True, quoting=csv.QUOTE_MINIMAL, index=None, **kwargs)


def read_tsv(fpath: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(fpath, sep='\t', header=0, quoting=csv.QUOTE_MINIMAL, **kwargs)
    return df


SplitsType = Union[int, list[int], tuple[int, ...], float, list[float], tuple[float, ...]]


def is_iter_len(x: object) -> bool:
    return hasattr(x, '__iter__') and hasattr(x, '__len__')


def split_range(n: int, splits: SplitsType) -> list[int]:
    if n == 0:
        return []
    negative = False
    if n < 0:
        negative = True
        n = -n
    def postproc(l: list[int]):
        if negative:
            return [-x for x in reversed(l)]
        return l

    res = [0]
    if type(splits) == int:
        if splits <= 0:
            return []
        div, rem = divmod(n, splits)
        if div == 0:
            splits = rem
        i_split, i = 0, 0
        while i_split < splits:
            off = div
            if rem > 0:
                off += 1
                rem -= 1
            i += off
            res.append(i)
            i_split += 1
        return postproc(res)

    isitlen = is_iter_len(splits)
    if isitlen and len(splits) == 0:
        return []

    if type(splits) == float:
        splits = [splits]
        isitlen = True

    if isitlen and any(type(s) == float for s in splits):
        has_int = any(type(s) == int for s in splits)
        spl = []
        was_neg, total = False, 0
        for s in splits:
            if s < 0:
                x = -1
                was_neg = True
            elif type(s) == float:
                x = int(n * s)
                assert x > 0, f'Zero sized split = {s} out of {n}'
            else:
                x = s
            spl.append(x)
            total += x

        if not was_neg and total < n and not has_int:
            spl.append(n - total)
        splits = spl

    if isitlen and type(splits[0]) == int:
        splits = [sp for sp in splits if sp != 0]
        if len(splits) == 0:
            return postproc([0, n])
        was_neg, total = False, 0
        for s in splits:
            assert type(s) == int
            if s == -1:
                assert not was_neg
                was_neg = True
            else:
                assert s > 0
                total += s
        assert was_neg and total < n or not was_neg and total == n, f'was_neg: {was_neg}. total: {total}. n: {n}'
        i, rest = 0, n - total
        for s in splits:
            i += (s if s > 0 else rest)
            res.append(i)
        return postproc(res)

    raise Exception(f'Unknown splits format: {splits}')


T = TypeVar('T')

def coalesce(val: Optional[T], fallback_val: T) -> T:
    if val is None:
        return fallback_val
    return val


def reraise(*args, **kwargs):
    raise


def bool_to_str(val: bool, first: bool = True, cap: bool = True) -> str:
    res = str(val)
    if first:
        res = res[0]
    if cap:
        res = res.upper()
    else:
        res = res.lower()
    return res


def rethrow(e):
    raise e


def instantiate_class(module_path: str, cls_name: str, *args: list[Any], **kwargs: Dict[str, Any]) -> object:
    module = import_module(module_path)
    cls = getattr(module, cls_name)
    instance = cls(*args, **kwargs)
    return instance


def instantiate_torch_optimizer(cls_name: str, params: Any, **kwargs) -> optim.Optimizer:
    opt_cls = getattr(optim, cls_name)
    optimizer = opt_cls(params, **kwargs)
    return optimizer


def instantiate_torch_lr_scheduler(cls_name: str, optimizer: optim.Optimizer, **kwargs) -> optim.lr_scheduler._LRScheduler:
    sched_cls = getattr(optim.lr_scheduler, cls_name)
    scheduler = sched_cls(optimizer, **kwargs)
    return scheduler
