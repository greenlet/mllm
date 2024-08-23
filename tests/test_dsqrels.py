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
            'qid': [],
            'query': [],
        }, dtype=int)
        df_off_1 = pd.DataFrame({
            'did': [],
            'offset': [],
        }, dtype=int)
        df_qrels_1 = pd.DataFrame({
            'qid': [],
            'did': [],
        }, dtype=int)
        df_qs_2 = pd.DataFrame({
            'qid': np.array([-1, -2, -3], dtype=int),
            'query': ['wow', 'hi', 'amazing view'],
        })
        df_off_2 = pd.DataFrame({
            'did': [111, 222, 333, 444],
            'offset': [0, 1000, 2300, 2600],
        }, dtype=int)
        df_qrels_2 = pd.DataFrame({
            'qid': [-1, -1, -2, -3, -3],
            'did': [111, 222, 333, 444, 333],
        }, dtype=int)
        df_qs, df_qrels, df_off = join_qrels_datasets(
            ds_ids, [df_qs_1, df_qs_2], [df_qrels_1, df_qrels_2], [df_off_1, df_off_2],
        )

        df_qs_1['dsid'] = 1
        df_qs_2['dsid'] = 2
        df_qs_exp = pd.concat([df_qs_1, df_qs_2], axis=0)
        df_qs_exp['dsqid'] = np.arange(len(df_qs_exp), dtype=int)
        df_off_1['dsid'] = 1
        df_off_2['dsid'] = 2
        df_off_exp = pd.concat([df_off_1, df_off_2], axis=0)
        df_off_exp['dsdid'] = np.arange(len(df_off_exp), dtype=int)
        df_qrels_1['dsid'] = 1
        df_qrels_2['dsid'] = 2
        df_qrels_exp = pd.concat([df_qrels_1, df_qrels_2], axis=0)
        df_qrels_exp['dsqid'] = np.array([0, 0, 1, 2, 2], dtype=int)
        df_qrels_exp['dsdid'] = np.array([0, 1, 2, 3, 2], dtype=int)

        self.assertTrue(df_qs.equals(df_qs_exp))
        self.assertTrue(df_off.equals(df_off_exp))
        self.assertTrue(df_qrels.equals(df_qrels_exp))

    def test_join_qrels_02(self):
        ds_ids = [1, 2]
        df_qs_1 = pd.DataFrame({
            'qid': np.array([1, 2, 3], dtype=int),
            'query': ['11', 'two', '-3'],
        })
        df_off_1 = pd.DataFrame({
            'did': [11, 22, 33],
            'offset': [0, 100, 230],
        }, dtype=int)
        df_qrels_1 = pd.DataFrame({
            'qid': [1, 1, 2, 3, 3, 3],
            'did': [11, 22, 33, 11, 22, 33],
        }, dtype=int)
        df_qs_2 = pd.DataFrame({
            'qid': np.array([-1, -2, -3], dtype=int),
            'query': ['wow', 'hi', 'amazing view'],
        })
        df_off_2 = pd.DataFrame({
            'did': [111, 222, 333, 444],
            'offset': [0, 1000, 2300, 2600],
        }, dtype=int)
        df_qrels_2 = pd.DataFrame({
            'qid': [-1, -1, -2, -3, -3],
            'did': [111, 222, 333, 444, 333],
        }, dtype=int)
        df_qs, df_qrels, df_off = join_qrels_datasets(
            ds_ids, [df_qs_1, df_qs_2], [df_qrels_1, df_qrels_2], [df_off_1, df_off_2],
        )

        df_qs_1['dsid'] = 1
        df_qs_2['dsid'] = 2
        df_qs_exp = pd.concat([df_qs_1, df_qs_2], axis=0)
        df_qs_exp['dsqid'] = np.arange(len(df_qs_exp), dtype=int)
        df_off_1['dsid'] = 1
        df_off_2['dsid'] = 2
        df_off_exp = pd.concat([df_off_1, df_off_2], axis=0)
        df_off_exp['dsdid'] = np.arange(len(df_off_exp), dtype=int)
        df_qrels_1['dsid'] = 1
        df_qrels_2['dsid'] = 2
        df_qrels_exp = pd.concat([df_qrels_1, df_qrels_2], axis=0)
        df_qrels_exp['dsqid'] = np.array([0, 0, 1, 2, 2, 2, 3, 3, 4, 5, 5], dtype=int)
        df_qrels_exp['dsdid'] = np.array([0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 5], dtype=int)

        # print(df_qrels)
        # print(df_qrels_exp)
        self.assertTrue(df_qs.equals(df_qs_exp))
        self.assertTrue(df_off.equals(df_off_exp))
        self.assertTrue(df_qrels.equals(df_qrels_exp))

    def _gen_offsets(self, n_docs: int) -> np.ndarray:
        off = np.random.randint(50, 1000, size=n_docs, dtype=int)
        off = np.cumsum(off)
        return off

    def _gen_qrels(self, query_ids: np.ndarray, doc_ids: np.ndarray,
                   ds_id: int, ds_query_ids: np.ndarray, ds_doc_ids: np.ndarray,
                   min_docs_per_query: int, max_docs_per_query: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        nq, nd = len(query_ids), len(doc_ids)
        assert nd >= min_docs_per_query
        assert max_docs_per_query >= min_docs_per_query
        max_docs_per_query = min(max_docs_per_query, nd)
        qid_res, did_res, dsid_res, dsqid_res, dsdid_res = [], [], [], [], []
        data = {'qid': qid_res, 'did': did_res}
        data_exp = {**data, 'dsid': dsid_res, 'dsqid': dsqid_res, 'dsdid': dsdid_res}
        if nd == 0:
            return pd.DataFrame(data, dtype=int), pd.DataFrame(data_exp, dtype=int)

        docs_per_query = np.random.randint(min_docs_per_query, max_docs_per_query + 1, size=nq)
        for query_id, ndocs, ds_query_id in zip(query_ids, docs_per_query, ds_query_ids):
            doc_inds = np.random.choice(nd, size=ndocs, replace=False)
            for doc_ind in doc_inds:
                qid_res.append(query_id)
                did_res.append(doc_ids[doc_ind])
                dsid_res.append(ds_id)
                dsqid_res.append(ds_query_id)
                dsdid_res.append(ds_doc_ids[doc_ind])

        return pd.DataFrame(data, dtype=int), pd.DataFrame(data_exp, dtype=int)

    def _gen_datasets(self, n_qs_l: list[int], n_docs_l: list[int],
                      min_docs_per_query_l: list[int], max_docs_per_query_l: list[int]) -> \
            tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame],
            pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert len(n_qs_l) == len(n_docs_l) == len(min_docs_per_query_l) == len(max_docs_per_query_l)
        n_dss = len(n_qs_l)
        ds_ids = np.arange(1, n_dss + 1)
        dfs_qs, dfs_off, dfs_qrels = [], [], []
        dfs_qs_exp, dfs_off_exp, dfs_qrels_exp = [], [], []
        qs_off, docs_off = 0, 0
        i = 1
        for ds_id, n_qs, n_docs, min_docs_per_query, max_docs_per_query in zip(ds_ids, n_qs_l, n_docs_l, min_docs_per_query_l, max_docs_per_query_l):
            query_ids = np.arange(n_qs, dtype=int)
            doc_ids = np.arange(n_docs, dtype=int)
            df_qs = pd.DataFrame({
                'qid': query_ids,
                'query': [str(query_id) * i for query_id in query_ids]
            })
            ds_query_ids = np.arange(qs_off, qs_off + n_qs, dtype=int)
            df_qs_exp = df_qs.copy()
            df_qs_exp['dsid'] = ds_id
            df_qs_exp['dsqid'] = ds_query_ids

            off = self._gen_offsets(n_docs)
            df_off = pd.DataFrame({
                'did': doc_ids,
                'offset': off,
            })
            ds_doc_ids = np.arange(docs_off, docs_off + n_docs, dtype=int)
            df_off_exp = df_off.copy()
            df_off_exp['dsid'] = ds_id
            df_off_exp['dsdid'] = ds_doc_ids

            df_qrels, df_qrels_exp = self._gen_qrels(query_ids, doc_ids, ds_id, ds_query_ids, ds_doc_ids, min_docs_per_query, max_docs_per_query)

            dfs_qs.append(df_qs)
            dfs_qs_exp.append(df_qs_exp)
            dfs_off.append(df_off)
            dfs_off_exp.append(df_off_exp)
            dfs_qrels.append(df_qrels)
            dfs_qrels_exp.append(df_qrels_exp)

            i += 1

        df_qs_exp = pd.concat(dfs_qs_exp, axis=0)
        df_off_exp = pd.concat(dfs_off_exp, axis=0)
        df_qrels_exp = pd.concat(dfs_qrels_exp, axis=0)

        return dfs_qs, dfs_off, dfs_qrels, df_qs_exp, df_off_exp, df_qrels_exp

    def test_join_qrels_03(self):
        ds_ids = [1, 2]

        dfs_qs, dfs_off, dfs_qrels, df_qs_exp, df_off_exp, df_qrels_exp = \
            self._gen_datasets([1000, 1500], [5000, 8000], [1, 0], [3, 1])

        df_qs, df_qrels, df_off = join_qrels_datasets(
            ds_ids, dfs_qs, dfs_qrels, dfs_off,
        )

        self.assertTrue(df_qs.equals(df_qs_exp))
        self.assertTrue(df_off.equals(df_off_exp))
        self.assertTrue(df_qrels.equals(df_qrels_exp))

