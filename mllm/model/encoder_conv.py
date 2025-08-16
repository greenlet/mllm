import re
import sys
from typing import Optional, Union

from mllm.config.model import EncoderConvCfg
from mllm.model.bert_generation.modeling_bert_generation import BertGenerationEmbeddings
from mllm.model.utils import get_top_vects
from mllm.train.utils import get_activation_module
from transformers import BertModel, PreTrainedModel, BertTokenizerFast

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class EncoderConv(nn.Module):
    cfg: EncoderConvCfg
    layers: nn.ModuleList

    def __init__(self, cfg: EncoderConvCfg):
        super().__init__()
        self.cfg = cfg

    # Tensor of integer tokens: [batch_size, seq_len]
    def forward(self, inp: Tensor) -> Tensor:
        batch_size, seq_len = inp.shape

