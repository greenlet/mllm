import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
import torch
import torch.utils.tensorboard as tb
from tqdm import trange


from mllm.model.model import CfgMllm, Mllm


class ArgsTrain(BaseModel):
    ds_file_path: Path = Field(
        None,
        required=False,
        description='Dataset CSV file path.',
        cli=('--ds-file-path',),
    )
    train_root_path: Path = Field(
        ...,
        required=True,
        description='Path to train root directory. New train processes will create subdirectories in it.',
        cli=('--train-root-path',),
    )
    batch_size: Optional[int] = Field(
        None,
        required=False,
        description='Batch size.',
        cli=('--batch-size',),
    )
    device: str = Field(
        'cpu',
        required=False,
        description='Device to run training on. Can have values: "cpu", "gpu"',
        cli=('--device',)
    )
    epochs: Optional[int] = Field(
        None,
        required=False,
        description='Number of training epochs.',
        cli=('--epochs',),
    )
    learning_rate: float = Field(
        0.001,
        required=False,
        description='Initial learning rate of the training process.',
        cli=('--learning-rate',)
    )
    train_epoch_steps: Optional[int] = Field(
        None,
        required=False,
        description='Number of training steps per epoch.',
        cli=('--train-epoch-steps',),
    )
    val_epoch_steps: Optional[int] = Field(
        None,
        required=False,
        description='Number of validation steps per epoch.',
        cli=('--val-epoch-steps',),
    )


def main(args: ArgsTrain) -> int:
    print(args)
    return 0


if __name__ == '__main__':
    run_and_exit(ArgsTrain, main, 'Train Mllm model.')


