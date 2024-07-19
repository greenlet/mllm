from datetime import datetime
from typing import Optional


DT_PAT = '%Y%m%d_%H%M%S'
DT_PAT_RE = r'\d{8}_\d{6}'


def gen_dt_str(dt: Optional[datetime] = None) -> str:
    dt = dt if dt is not None else datetime.now()
    return dt.strftime(DT_PAT)


def parse_dt_str(dt_str: str, silent: bool = True) -> Optional[datetime]:
    try:
        return datetime.strptime(dt_str, DT_PAT)
    except:
        pass

