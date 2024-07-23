
import numpy as np
import torch
from torch import nn, Tensor

from mllm.model.config import CfgMllmEncdec
from mllm.model.modules import EncoderLayer, VocabEncoder, EmbDecoder, VocabDecoder, Encoder


class MllmEncdec(nn.Module):
    cfg: CfgMllmEncdec
    vocab_encoder: VocabEncoder
    encoder: Encoder
    decoder: EmbDecoder
    vocab_decoder: VocabDecoder

    def __init__(self, cfg: CfgMllmEncdec):
        super().__init__()
        self.cfg = cfg.copy(deep=True)
        self.vocab_encoder = VocabEncoder(
            **cfg.vocab_encoder.dict(),
        )
        self.encoder = Encoder(**self.cfg.encoder.dict())
        self.decoder = EmbDecoder(**self.cfg.decoder.dict())
        self.vocab_decoder = VocabDecoder(
            d_model=cfg.encoder.d_model,
            n_vocab=cfg.vocab_encoder.n_vocab,
        )

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
        return self.vocab_encoder(inp)

    def run_encoder(self, inp: Tensor) -> tuple[Tensor, Tensor]:
        out = self.encoder(inp)[0]
        if self.cfg.encoder.with_emb_mat:
            out_seq = out_emb = out
        else:
            out_seq, out_emb = out[..., :-1, :], out[..., -1, :]
        return out_seq, out_emb

    def run_decoder(self, inp: Tensor) -> Tensor:
        return self.decoder(inp)

    def run_vocab_decoder(self, inp: Tensor) -> Tensor:
        return self.vocab_decoder(inp)
    
    def run_enc_emb(self, inp: Tensor) -> Tensor:
        out = self.run_vocab_encoder(inp)
        out = self.run_encoder(out)
        return out[1]
    
    def forward(self, inp_chunks: Tensor) -> Tensor:
        out_enc_0 = self.run_vocab_encoder(inp_chunks)
        _, out_enc_1 = self.run_encoder(out_enc_0)

        out_dec_0 = self.run_decoder(out_enc_1)
        out_dec_logits = self.run_vocab_decoder(out_dec_0)

        return out_dec_logits


