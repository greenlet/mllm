import unittest as ut

import numpy as np

from mllm.utils.utils import split_range


class TestSplitRange(ut.TestCase):
    def setUp(self):
        pass

    def test_split_range_01(self):
        def exc_split(n: int, split: float, err_only: bool = False):
            err = f'Zero sized split = {split} out of {n}'
            if err_only:
                return err
            return n, split, err

        def exc_total(n: int, split: list, was_neg: bool, total: int):
            return n, split, f'was_neg: {was_neg}. total: {total}. n: {n}'

        for n, splits, exp in [
            (0, 1, []),
            (1, 0, []),
            (1, 1, [0, 1]),
            (1, 2, [0, 1]),
            (2, 2, [0, 1, 2]),
            (7, 5, [0, 2, 4, 5, 6, 7]),
            (-8, 2, [-8, -4, 0]),
            (-3, 0, []),
            (-3, 1, [-3, 0]),
            (0, 0.5, []),
            exc_split(1, 0.5),
            exc_split(1, 0.0),
            (2, 0.5, [0, 1, 2]),
            exc_split(2, 0.4),
            (6, 0.3, [0, 1, 6]),
            (6, [0.3, -1], [0, 1, 6]),
            (8, 0.5, [0, 4, 8]),
            (8, [0.5, -1], [0, 4, 8]),
            (10, [0.3, 0.3], [0, 3, 6, 10]),
            (10, [0.3, 0.3, -1], [0, 3, 6, 10]),
            exc_total(10, [0.3, 0.3, 0.5], False, 11),
            exc_total(10, [0.3, 0.7, -1], True, 10),
            (40, [0.25, 0.25], [0, 10, 20, 40]),
            exc_total(10, [1, 7], False, 8),
            exc_total(10, [5, 7, -1], True, 12),
            (10, [1, 7, -1], [0, 1, 8, 10]),
            (15, [3, 8, 4], [0, 3, 11, 15]),
            (15, [5, 0.2, -1], [0, 5, 8, 15]),
            exc_total(15, [5, 0.2], False, 8),
            (15, [5, 0.06], exc_split(15, 0.06, True)),
            (15, [5, 0.06, -1], exc_split(15, 0.06, True)),
            (15, [5, 0.07, -1], [0, 5, 6, 15]),
            (15, [0.3, 0.06, -1], exc_split(15, 0.06, True)),
            (-15, [5, 0.07, -1], [-15, -6, -5, 0]),
            exc_total(10000, [100, 0.2, 0.8], False, 10100),
            exc_total(10000, [100, 0.2], False, 2100),
            exc_total(10000, [5000, 0.1, 0.4, -1], True, 10000),
            (10000, [5000, 0.1, -1], [0, 5000, 6000, 10000]),
            (10000, [5000, 0.1, 0.4], [0, 5000, 6000, 10000]),
        ]:
            try:
                res = split_range(n, splits)
            except Exception as e:
                res = str(e)
            def msg() -> str:
                return f'n = {n}. splits = {splits}\nresult:\n    {res}\nexpected:\n    {exp}'
            self.assertTrue(res == exp, msg())

