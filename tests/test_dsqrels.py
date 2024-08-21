import unittest as ut

import numpy as np
import pandas as pd

from mllm.data.dsqrels import join_qrels_datasets


class TestDsQrels(ut.TestCase):
    def setUp(self):
        pass

    def test_join_qrels_01(self):
        ds_ids = [1, 2]
        df_qs_1 = pd.DataFrame({
            'qid': [1, 2, 3],
            'query': ['11', 'two', '-3'],
        })
        df_off_1 = pd.DataFrame({
            'did': [11, 22, 33],
            'offset': [0, 100, 230],
        })
        df_qrels_1 = pd.DataFrame({
            'qid': [1, 1, 2, 3, 3, 3],
            'did': [11, 22, 33, 11, 22, 33],
        })
        df_qs_2 = pd.DataFrame({
            'qid': [-1, -2, -3],
            'query': ['wow', 'hi', 'amazing view'],
        })
        df_off_2 = pd.DataFrame({
            'did': [111, 222, 333, 444],
            'offset': [0, 1000, 2300, 2600],
        })
        df_qrels_2 = pd.DataFrame({
            'qid': [-1, -1, -2, -3, -3],
            'did': [111, 222, 333, 444, 333],
        })
        df_qs, df_qrels, df_off = join_qrels_datasets(
            ds_ids, [df_qs_1, df_qs_2], [df_qrels_1, df_qrels_2], [df_off_1, df_off_2],
        )

        df_qs_1['dsid'] = 1
        df_qs_2['dsid'] = 2
        df_qs_exp = pd.concat([df_qs_1, df_qs_2], axis=0)
        df_qs_exp['dsqid'] = np.arange(len(df_qs_exp))
        df_off_1['dsid'] = 1
        df_off_2['dsid'] = 2
        df_off_exp = pd.concat([df_off_1, df_off_2], axis=0)
        df_off_exp['dsdid'] = np.arange(len(df_off_exp))
        df_qrels_1['dsid'] = 1
        df_qrels_2['dsid'] = 2
        df_qrels_exp = pd.concat([df_qrels_1, df_qrels_2], axis=0)
        df_qrels_exp['dsqid'] = [0, 0, 1, 2, 2, 2, 3, 3, 4, 5, 5]
        df_qrels_exp['dsdid'] = [0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 5]

        self.assertEqual(df_qs, df_qs_exp)
        self.assertEqual(df_off, df_off_exp)
        self.assertEqual(df_qrels, df_qrels_exp)

