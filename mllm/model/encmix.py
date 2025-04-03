import re
import sys
from typing import Optional

from transformers import BertModel, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from mllm.config.model import EncmixBertCfg
from mllm.model.mix_bert import MixBertModel

if '..' not in sys.path: sys.path.append('..')

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F



class EncmixBert(nn.Module):
    cfg: EncmixBertCfg
    tkz: PreTrainedTokenizer
    device: torch.device
    bert_model: MixBertModel

    def __init__(self, cfg: EncmixBertCfg, tkz: PreTrainedTokenizer, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg
        self.tkz = tkz
        self.device = device if device is not None else torch.device('cpu')
        self.bert_model = MixBertModel.from_pretrained(
            self.cfg.pretrained_model_name, torch_dtype=torch.float32, device_map=self.device,
        )
        print(self.bert_model)

    # chunk_toks: [n_chunks, seq_len]
    # plain_toks: [n_plain_toks]
    # target_toks: [n_target_toks]
    def run_chunks_plain_seq(self, chunk_toks: torch.Tensor, plain_toks: torch.Tensor, target_toks: Optional[torch.Tensor] = None) -> torch.Tensor:
        n_chunks = chunk_toks.shape[0]
        chunk_toks_mask = chunk_toks != self.tkz.pad_token_id
        # [n_chunks, seq_len] -> [n_chunks, seq_len, d_model]
        chunks_out: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
            input_ids=chunk_toks, attention_mask=chunk_toks_mask
        )
        # [n_chunks, seq_len, d_model] -> [n_chunks, d_model]
        chunks_emb = chunks_out.last_hidden_state[: ,0]
        # [n_chunks, d_model] -> [1, n_chunks, d_model]
        chunks_emb = chunks_emb.unsqueeze(0)

        # [n_plain_toks] -> [1, n_plain_toks]
        plain_toks = plain_toks.unsqueeze(0)

        mix_mask_emb = torch.ones_like(chunks_emb, dtype=torch.bool, device=self.device)
        mix_mask_plain = target_toks > 0
        mix_mask = torch.concatenate([mix_mask_emb, mix_mask_plain])
        mix_out: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
            inputs_starting_embeds=chunks_emb, input_ids=plain_toks, attention_mask=mix_mask,
        )


