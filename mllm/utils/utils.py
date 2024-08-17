import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


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

