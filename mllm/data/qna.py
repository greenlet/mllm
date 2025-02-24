import json
import os
import sys
from pathlib import Path
import requests

import pandas as pd


DATA_PATH = Path(os.path.expandvars('$HOME')) / 'data'
HOTPOTQA_URL = 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json'
HOTPOTQA_DATA_PATH = DATA_PATH / 'hotpotqa'


def get_hotpotqa(url: str = HOTPOTQA_URL, dir_path: Path = HOTPOTQA_DATA_PATH) -> pd.DataFrame:
    fname = url.split('/')[-1]
    fpath = dir_path / fname
    if fpath.exists():
        print(f'Load {fpath}')
        data = json.loads(fpath.read_text())
    else:
        print(f'Load {url}')
        resp = requests.get(url)
        data = resp.json()
        print(f'Write {fpath}')
        fpath.write_text(resp.text)
    df = pd.DataFrame.from_records(data)
    return df




