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
        assert self.tkz.pad_token_id == 0, f'pad_token_id = {self.tkz.pad_token_id}'

    # chunk_toks: [n_chunks, seq_len]
    # plain_toks: [n_plain_toks]
    # target_toks: [n_target_toks]
    def run_chunks_plain_seq(self, chunk_toks: torch.Tensor, plain_toks: Optional[torch.Tensor] = None, target_toks: Optional[torch.Tensor] = None) -> torch.Tensor:
        n_chunks = chunk_toks.shape[0]
        chunk_toks_mask = chunk_toks != self.tkz.pad_token_id
        # [n_chunks, seq_len] -> [n_chunks, seq_len, d_model]
        chunks_out: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
            input_ids=chunk_toks, attention_mask=chunk_toks_mask
        )
        # [n_chunks, seq_len, d_model] -> [n_chunks, d_model]
        chunks_emb = chunks_out.last_hidden_state[: ,0]

        if target_toks is None:
            target_toks = torch.tensor([self.tkz.mask_token_id], dtype=plain_toks.dtype, device=self.device)
        n_target_toks = len(target_toks)

        # [n_target_toks] -> [n_target_toks, n_target_toks]
        target_toks_inp = target_toks.repeat(n_target_toks, 1)
        target_toks_inp = torch.tril(target_toks_inp)
        target_mask = torch.eye(n_target_toks, dtype=torch.bool)
        target_toks_inp[target_mask] = self.tkz.mask_token_id

        # [n_chunks, d_model] -> [n_target_toks, n_chunks, d_model]
        chunks_emb = chunks_emb.repeat(n_target_toks, 1, 1)

        if plain_toks is None:
            n_plain_toks = 0
            toks_inp = target_toks_inp
        else:
            n_plain_toks = len(plain_toks)
            # [n_plain_toks] -> [n_target_toks, n_plain_toks]
            plain_toks_inp = plain_toks.repeat(n_target_toks, 1, 1)
            # [n_target_toks, n_plain_toks], [n_target_toks, n_target_toks] -> [n_target_toks, n_plain_toks + n_target_toks]
            toks_inp = torch.concatenate([plain_toks_inp, target_toks_inp], dim=1)

        toks_inp_mask = toks_inp != self.tkz.pad_token_id
        chunks_mask = torch.ones((n_target_toks, n_chunks), dtype=torch.bool, device=self.device)
        inp_mask = np.concatenate([chunks_mask, toks_inp_mask])
        mix_out: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
            inputs_starting_embeds=chunks_emb, input_ids=toks_inp, attention_mask=inp_mask,
        )

        out_logits = mix_out.last_hidden_state[:, n_chunks + n_plain_toks:][target_mask]
        return out_logits


