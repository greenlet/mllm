from typing import Union, TypeVar

import numpy as np
import torch
from torch import nn, Tensor

from mllm.model.config import CfgMllmRanker, create_mllm_ranker_cfg
from mllm.model.modules import VocabEncoder, Encoder, Decoder


class MllmRanker(nn.Module):
    cfg: CfgMllmRanker
    # vocab_encoder: VocabEncoder
    # encoders: nn.ModuleList
    # decoders: nn.ModuleList

    def __init__(self, cfg: CfgMllmRanker):
        super().__init__()
        self.cfg = cfg.copy(deep=True)
        self.vocab_encoder = VocabEncoder(
            **cfg.vocab_encoder.dict(),
        )
        self.encoders = nn.ModuleList([
            Encoder(**cfg_enc.dict()) for cfg_enc in cfg.encoders
        ])
        self.decoders = nn.ModuleList([
            Decoder(**cfg_dec.dict()) for cfg_dec in cfg.decoders
        ])
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

    def run_encoder(self, level_num: int, inp: Tensor) -> tuple[Tensor, Tensor]:
        ind = level_num - 1
        out = self.encoders[ind](inp)[0]
        if self.cfg.encoders[ind].with_emb_mat:
            out_seq = out_emb = out
        else:
            out_seq, out_emb = out[..., :-1, :], out[..., -1, :]
        return out_seq, out_emb

    def run_decoder(self, level_num: int, inp: Tensor) -> Tensor:
        ind = level_num - 1
        return self.decoders[ind](inp)[0]

    def forward(self, target_chunks: Tensor, docs_chunks: Tensor) -> Tensor:
        level_num = 1
        n_target = target_chunks.shape[0]
        inp_chunks = torch.concat((docs_chunks, target_chunks), dim=0)
        out_enc_0 = self.run_vocab_encoder(inp_chunks)
        _, out_enc_1 = self.run_encoder(level_num, out_enc_0)
        out_enc_1 = out_enc_1.unsqueeze(0)

        out_dec_0 = self.run_decoder(level_num, out_enc_1)

        out_dec_rank = out_dec_0[:, :-n_target]
        return out_dec_rank

    def forward_1(self, level_num: int, target_chunks: Tensor, docs_chunks: Tensor) -> Tensor:
        n_target = target_chunks.shape[0]
        inp_chunks = torch.concat((target_chunks, docs_chunks), dim=0)
        out_enc_0 = self.run_vocab_encoder(inp_chunks)
        _, out_enc_1 = self.run_encoder(level_num, out_enc_0)
        out_enc_1 = out_enc_1.unsqueeze(0)

        pad_inp = np.array([[self.cfg.vocab_encoder.pad_idx]], dtype=np.int32)
        pad_inp = torch.from_numpy(pad_inp).to(target_chunks.device)
        pad_emb = self.vocab_encoder.src_word_emb(pad_inp)
        out_enc_1_tgt, out_enc_1_doc = out_enc_1[:, :n_target], out_enc_1[:, n_target:]
        out_enc_1 = torch.concat([out_enc_1_tgt, pad_emb, out_enc_1_doc], dim=1)

        out_dec_0 = self.run_decoder(level_num, out_enc_1)
        out_dec_rank = out_dec_0[:, n_target + 1:]
        return out_dec_rank


def test_create_mllm_ranker():
    cfg_mllm = create_mllm_ranker_cfg(n_vocab=50_000)
    print(cfg_mllm)
    mllm = MllmRanker(cfg_mllm)
    print(mllm)


if __name__ == '__main__':
    test_create_mllm_ranker()

