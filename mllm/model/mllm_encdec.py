from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from mllm.config.model import MllmEncdecCfg, EncoderCfg, EmbDecoderCfg
from mllm.model.modules import EncoderLayer, VocabEncoder, EmbDecoder, VocabDecoder, Encoder


# embs_pred [n_batch, seq_len, emb_size] float32 - embeddings sequence predicted by model
# embs_gt [n_batch, seq_len, emb_size] float32 - ground truth embeddings
def encdec_embs_loss_cos(embs_pred: torch.Tensor, embs_gt: torch.Tensor) -> torch.Tensor:
    # [n_batch, seq_len]
    cos_sim = F.cosine_similarity(embs_pred, embs_gt, dim=-1)
    # []
    loss = 1 - torch.mean(cos_sim)
    return loss


class MllmEncdecLevel(nn.Module):
    cfg: MllmEncdecCfg
    level: int
    cfg_enc: EncoderCfg
    cfg_dec: EmbDecoderCfg
    encoder: Encoder
    decoder: EmbDecoder
    # vocab_encoder: Optional[VocabEncoder] = None
    # vocab_decoder: Optional[VocabDecoder] = None

    def __init__(self, cfg: MllmEncdecCfg, level: int):
        super().__init__()
        self.cfg = cfg.copy(deep=True)
        self.level = level
        self.cfg_enc = self.cfg.encoders[self.level]
        self.cfg_dec = self.cfg.decoders[self.level]
        self.vocab_encoder = None
        self.vocab_decoder = None
        if self.level == 0:
            self.vocab_encoder = VocabEncoder(**cfg.vocab_encoder.dict())
            if self.cfg.with_vocab_decoder:
                self.vocab_decoder = VocabDecoder(
                    d_model=self.cfg_enc.d_model,
                    n_vocab=cfg.vocab_encoder.n_vocab,
                )
        self.encoder = Encoder(**self.cfg_enc.dict())
        self.decoder = EmbDecoder(**self.cfg_dec.dict())

        for n, p in self.named_parameters():
            # if n == 'vocab_encoder.src_word_emb.weight':
            #     nn.init.normal_(p, std=0.1)
            # elif p.squeeze().dim() > 1:
            #     nn.init.xavier_normal_(p)
            # else:
            #     nn.init.normal_(p, std=0.1)
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, -0.1, 0.1)
            pnp = p.detach().cpu().numpy()
            print(n, pnp.shape, pnp.min(), pnp.mean(), pnp.max())

    def run_vocab_encoder(self, inp: Tensor) -> Tensor:
        out = inp
        if self.vocab_encoder is not None:
            out = self.vocab_encoder(inp)
        return out

    def run_encoder(self, inp: Tensor) -> tuple[Tensor, Tensor]:
        out = self.encoder(inp)[0]
        out_seq = out_emb = out
        # if self.cfg_enc.with_emb_mat:
        #     out_seq = out_emb = out
        # else:
        #     out_seq, out_emb = out[..., :-1, :], out[..., -1, :]
        return out_seq, out_emb

    def run_decoder(self, inp: Tensor) -> Tensor:
        return self.decoder(inp)

    def run_vocab_decoder(self, inp: Tensor) -> Tensor:
        out = inp
        if self.vocab_decoder is not None:
            out = self.vocab_decoder(inp)
        return out
    
    def run_enc_emb(self, inp: Tensor) -> Tensor:
        out = self.run_vocab_encoder(inp)
        out = self.run_encoder(out)
        return out[1]
    
    def forward(self, inp_chunks: Tensor) -> Tensor:
        out_enc_0 = self.run_vocab_encoder(inp_chunks)
        _, out_enc_1 = self.run_encoder(out_enc_0)

        out_dec_0 = self.run_decoder(out_enc_1)
        out_dec_1 = self.run_vocab_decoder(out_dec_0)

        return out_dec_1


