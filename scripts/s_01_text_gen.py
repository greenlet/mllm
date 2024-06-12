import os
from pathlib import Path
import sys

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from transformers import pipeline, set_seed
import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions


generator = pipeline('text-generation', model='gpt2')
set_seed(42)
res = generator('Paris city ', max_length=30, num_return_sequences=5)
print(res)
