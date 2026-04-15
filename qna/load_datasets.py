"""Pre-download QnA datasets in parallel using multiprocessing.

Usage:
    python -m qna.load_datasets --data-path ./data --qna-datasets squad_v2 natural_questions triviaqa
"""

import argparse
import multiprocessing as mp
import sys
import traceback
from pathlib import Path
from typing import List

from mllm.data.qna.dataset import QnaDatasetType, QNA_DATASETS_DEFAULT


# Maps enum value → (module_path, loader_function_name)
_LOADER_REGISTRY = {
    QnaDatasetType.SQUAD_V2: ('mllm.data.qna.ds_01_squad_v2', 'load_squad_v2'),
    QnaDatasetType.NATURAL_QUESTIONS: ('mllm.data.qna.ds_02_natural_questions', 'load_nq'),
    QnaDatasetType.TRIVIAQA: ('mllm.data.qna.ds_03_triviaqa', 'load_triviaqa'),
    QnaDatasetType.NEWSQA: ('mllm.data.qna.ds_04_newsqa', 'load_newsqa'),
    QnaDatasetType.QUAC: ('mllm.data.qna.ds_05_quac', 'load_quac'),
    QnaDatasetType.COQA: ('mllm.data.qna.ds_06_coqa', 'load_coqa'),
    QnaDatasetType.MRQA: ('mllm.data.qna.ds_07_mrqa', 'load_mrqa'),
    QnaDatasetType.ADVERSARIALQA: ('mllm.data.qna.ds_08_adversarialqa', 'load_adversarialqa'),
    QnaDatasetType.SQUAD_V1: ('mllm.data.qna.ds_09_squad_v1', 'load_squad_v1'),
}


def _load_one(args: tuple) -> str:
    """Worker function: import the loader and call it with cache_dir.

    Returns a status string.
    """
    ds_type_value, cache_dir = args
    ds_type = QnaDatasetType(ds_type_value)
    module_path, func_name = _LOADER_REGISTRY[ds_type]
    try:
        import importlib
        mod = importlib.import_module(module_path)
        loader = getattr(mod, func_name)
        loader(cache_dir=cache_dir)
        return f'OK: {ds_type.value}'
    except Exception:
        return f'FAIL: {ds_type.value}\n{traceback.format_exc()}'


def main():
    all_values = [t.value for t in QnaDatasetType]
    default_values = [t.value for t in QNA_DATASETS_DEFAULT]

    parser = argparse.ArgumentParser(description='Pre-download QnA datasets in parallel.')
    parser.add_argument(
        '--data-path', type=Path, required=True,
        help='Root directory for HuggingFace cache (passed as cache_dir to load_dataset).',
    )
    parser.add_argument(
        '--qna-datasets', nargs='+', type=str, choices=all_values,
        default=default_values,
        help=f'QnA datasets to download. Default: {" ".join(default_values)}',
    )
    args = parser.parse_args()

    cache_dir: Path = args.data_path
    datasets: List[str] = args.qna_datasets

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f'Cache directory: {cache_dir.resolve()}')
    print(f'Datasets to load ({len(datasets)}): {", ".join(datasets)}')

    worker_args = [(ds_val, str(cache_dir)) for ds_val in datasets]

    n_workers = min(len(worker_args), mp.cpu_count() or 1)
    print(f'Starting {n_workers} workers ...')

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_load_one, worker_args)

    print('\n' + '=' * 60)
    failures = []
    for r in results:
        print(r)
        if r.startswith('FAIL'):
            failures.append(r)
    print('=' * 60)

    if failures:
        print(f'\n{len(failures)} dataset(s) failed to load.')
        sys.exit(1)
    else:
        print(f'\nAll {len(results)} dataset(s) loaded successfully.')


if __name__ == '__main__':
    main()
