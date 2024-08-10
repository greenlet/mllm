import os.path
from pathlib import Path

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as, to_yaml_file


class A(BaseModel):
    x: int

class B(BaseModel):
    x: float

def test_pydantic_yaml():
    dpath = os.path.expandvars('$HOME/data/tmp')
    dpath = Path(dpath)
    dpath.mkdir(exist_ok=True)
    fpath = dpath / 'test.yaml'
    a = A(x=1)
    to_yaml_file(fpath, a)
    a1 = parse_yaml_file_as(A, fpath)
    print(a1)
    a2 = parse_yaml_file_as(B, fpath)
    print(a2)





if __name__ == '__main__':
    test_pydantic_yaml()

