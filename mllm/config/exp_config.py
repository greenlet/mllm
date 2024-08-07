from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field


class ExpConfig(BaseModel):
    version: str = '0.0.1'
    description: str = ''


class MllmRankerQsTrainCfg(ExpConfig):
    ds_dir_path: Path
    train_root_path: Path
    train_subdir: str
    docs_batch_size: int
    max_chunks_per_doc: int


class TrainMllmRankerQsCfg(ExpConfig):
    pass

