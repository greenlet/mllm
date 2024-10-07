from typing import Generator, Optional, TypeVar, Generic, Callable

import numpy as np

from mllm.utils.utils import SplitsType, split_range

TDs = TypeVar('TDs')
TBatch = TypeVar('TBatch')
TGetBatchFunc = Callable[[np.ndarray, ...], TBatch]


class DsView(Generic[TDs, TBatch]):
    ds: TDs
    ids: np.ndarray
    get_batch_fn: TGetBatchFunc
    batch_size: Optional[int] = None
    kwargs: dict

    def __init__(self, ds: TDs, ids: np.ndarray, get_batch_fn: TGetBatchFunc, batch_size: Optional[int] = None, **kwargs):
        self.ds = ds
        self.ids = ids.copy()
        self.get_batch_fn = get_batch_fn
        self.batch_size = batch_size
        self.kwargs = kwargs or {}

    def split(self, splits: SplitsType) -> tuple['DsView', ...]:
        intervals = split_range(len(self.ids), splits)
        res = []
        for i in range(1, len(intervals)):
            ids = self.ids[intervals[i - 1]:intervals[i]]
            view = DsView(
                ds=self.ds, ids=ids, get_batch_fn=self.get_batch_fn, batch_size=self.batch_size, **self.kwargs,
            )
            res.append(view)
        return tuple(res)

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def get_batch_iterator(self, n_batches: Optional[int] = None, batch_size: Optional[int] = None,
                           drop_last: bool = False, shuffle_between_loops: bool = True, **kwargs)\
            -> Generator[TBatch, None, None]:
        kwargs = {**self.kwargs, **kwargs}
        batch_size = batch_size or self.batch_size
        n = len(self.ids)
        n_batches_total = n // batch_size + min(n % batch_size, 1)

        info = f'n = {n}. batch_size = {batch_size}. n_batches = {n_batches}. n_batches_total = {n_batches_total}'
        assert n_batches_total > 0, info
        assert n_batches is None or n_batches > 0, info

        looped = False
        if n_batches is None:
            n_batches = n_batches_total
        if n_batches > n_batches_total:
            looped = True

        for i_batch in range(n_batches):
            if i_batch > 0 and i_batch % n_batches_total == 0:
                if shuffle_between_loops:
                    self.shuffle()
            i = (i_batch % n_batches_total) * batch_size
            batch_size_cur = min(batch_size, n - i)
            inds = range(i, i + batch_size_cur)
            if batch_size_cur < batch_size:
                if not looped:
                    if drop_last:
                        return
                else:
                    rest = batch_size - batch_size_cur
                    inds = list(range(i, n)) + list(range(rest))
            ids = self.ids[inds]
            batch = self.get_batch_fn(ids, **kwargs)
            yield batch

    def __len__(self) -> int:
        return len(self.ids)

    def shuffle(self):
        print(f'Shuffle {len(self.ids)} elements')
        np.random.shuffle(self.ids)




