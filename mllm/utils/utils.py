from datetime import datetime
from typing import Optional


DT_PAT = '%Y%m%d_%H%M%S'


def gen_dt_str(dt: Optional[datetime] = None) -> str:
    dt = dt if dt is not None else datetime.now()
    return dt.strftime(DT_PAT)

