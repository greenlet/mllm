
from enum import Enum
from typing import Generator, Optional

import numpy as np


class QrelsBatch:
    pass


class DsQrelsView:
    ds: 'DsQrels'
    ids: np.ndarray
    batch_size: Optional[int] = None

    def __init__(self, ds: 'DsQrels', ids: np.ndarray, batch_size: Optional[int] = None):
        self.ds = ds
        self.ids = ids
        self.batch_size = batch_size

    def split(self, first_ratio: float) -> tuple['DsQrelsView', 'DsQrelsView']:
        pass

    def get_batch_it(self, batch_size: Optional[None]) -> Generator[QrelsBatch, None, None]:
        pass


class DsQrels:
    ds_id: str
    qs_ids: np.ndarray

    def __init__(self, ds_id: str, qs_ids: np.ndarray):
        self.ds_id = ds_id
        self.qs_ids = qs_ids

    def get_view(self) -> DsQrelsView:
        pass

    @staticmethod
    def join(dss: list['DsQrels']) -> 'DsQrels':
        pass
