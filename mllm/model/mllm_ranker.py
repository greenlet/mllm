from typing import Union, TypeVar, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor

from mllm.config.model import MllmRankerCfg, create_mllm_ranker_cfg, EncoderCfg, VocabEncoderCfg
from mllm.model.modules import VocabEncoder, Encoder, Decoder, DecoderRankSimple


class RankProbLoss(nn.Module):
    def __init__(self, target_weight: float = 0.5):
        super().__init__()
        self.target_weight = target_weight
        self.register_buffer('prob_cap', torch.scalar_tensor(1e-6))

    def forward(self, prob_pred: list[torch.Tensor], mask_gt: Union[torch.Tensor, list[torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_tgt = torch.scalar_tensor(0, dtype=torch.float32, device=prob_pred[0].device)
        loss_nontgt = torch.scalar_tensor(0, dtype=torch.float32, device=prob_pred[0].device)
        n_batch = len(prob_pred)
        for i in range(n_batch):
            prob_tgt = torch.masked_select(prob_pred[i], mask_gt[i])
            prob_nontgt = 1 - torch.masked_select(prob_pred[i], ~mask_gt[i])

            prob_tgt = torch.maximum(prob_tgt, self.prob_cap)
            prob_nontgt = torch.maximum(prob_nontgt, self.prob_cap)
            loss_tgt += -torch.mean(torch.log(prob_tgt))
            loss_nontgt += -torch.mean(torch.log(prob_nontgt))

            # loss_tgt += 1 - torch.mean(prob_tgt)
            # loss_nontgt += 1 - torch.mean(prob_nontgt)

        loss_tgt /= n_batch
        loss_nontgt /= n_batch
        loss = self.target_weight * loss_tgt + (1 - self.target_weight) * loss_nontgt
        return loss, loss_tgt, loss_nontgt

    def forward_1(self, prob_pred: torch.Tensor, mask_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prob_pred = prob_pred.squeeze()
        prob_tgt = torch.masked_select(prob_pred, mask_gt)
        prob_nontgt = 1 - torch.masked_select(prob_pred, ~mask_gt)
        prob_tgt = torch.maximum(prob_tgt, self.prob_cap)
        prob_nontgt = torch.maximum(prob_nontgt, self.prob_cap)
        loss_tgt = -torch.mean(torch.log(prob_tgt))
        loss_nontgt = -torch.mean(torch.log(prob_nontgt))
        loss = self.target_weight * loss_tgt + (1 - self.target_weight) * loss_nontgt
        return loss, loss_tgt, loss_nontgt


class MllmRanker(nn.Module):
    cfg: MllmRankerCfg
    # vocab_encoder: VocabEncoder
    # encoders: nn.ModuleList
    # decoders: nn.ModuleList

    def __init__(self, cfg: MllmRankerCfg):
        super().__init__()
        self.cfg = cfg.copy(deep=True)
        self.vocab_encoder = VocabEncoder(
            **cfg.vocab_encoder.dict(),
        )
        self.encoders = nn.ModuleList([
            Encoder(**cfg_enc.dict()) for cfg_enc in cfg.encoders
        ])
        # self.decoders = nn.ModuleList([
        #     Decoder(**cfg_dec.dict()) for cfg_dec in cfg.decoders
        # ])
        self.decoders = nn.ModuleList([
            DecoderRankSimple(cfg.decoders[0].d_model)
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

    def run_enc_emb(self, inp: Tensor) -> Tensor:
        out = self.run_vocab_encoder(inp)
        out = self.run_encoder(1, out)
        return out[1]

    @staticmethod
    def _permute_chunks(chunks: Tensor, chunks_off_len: list[tuple[int, int]], i_target: int) -> tuple[Tensor, Tensor]:
        inds = torch.randperm(len(chunks_off_len))
        res = [chunks[i] for i in inds]
        return inds, res

    def run_qs(self, docs_chunks: Tensor, qs_chunks: Tensor, docs_off_len: list[tuple[int, int]],
               qs_off_len: list[tuple[int, int]]) -> tuple[list[Tensor], list[Tensor]]:
        n_docs, n_qs = len(docs_off_len), len(qs_off_len)
        assert n_docs == n_qs, f'# of docs ({n_docs}) != # of queries ({n_qs})'
        device = docs_chunks.device
        inp_chunks = torch.concat((docs_chunks, qs_chunks), dim=0)
        out_enc = self.run_vocab_encoder(inp_chunks)
        _, out_enc = self.run_encoder(1, out_enc)
        n_docs_chunks = docs_chunks.shape[0]
        docs_enc, qs_enc = out_enc[:n_docs_chunks], out_enc[n_docs_chunks:]

        qs_encs = []
        for query_off, query_len in qs_off_len:
            qs_encs.append(qs_enc[query_off:query_off + query_len])

        docs_encs = docs_enc.unsqueeze(0)
        ranks, masks = [], []
        for i_query in range(n_qs):
            query_enc = qs_encs[i_query].unsqueeze(0)
            out_rank = self.decoders[0](docs_encs, query_enc)
            ranks.append(out_rank)
            mask = torch.full((n_docs_chunks,), False, dtype=torch.bool, device=device)
            doc_off, doc_len = docs_off_len[i_query]
            mask[doc_off:doc_off + doc_len] = True
            masks.append(mask)

        return ranks, masks

    def run_qs_infer(self, docs_chunks: Tensor, qs_chunks: Tensor) -> Tensor:
        device = docs_chunks.device
        inp_chunks = torch.concat((docs_chunks, qs_chunks), dim=0)
        out_enc = self.run_vocab_encoder(inp_chunks)
        _, out_enc = self.run_encoder(1, out_enc)
        n_docs_chunks = docs_chunks.shape[0]
        docs_enc, qs_enc = out_enc[:n_docs_chunks], out_enc[n_docs_chunks:]

        out_rank = self.decoders[0](docs_enc.unsqueeze(0), qs_enc.unsqueeze(0))
        return out_rank

    # df_docs_ids. doc_emb_id: int (index), ds_id: int, ds_doc_id: int
    # df_qs_ids. query_emb_id: int (index), ds_id: int, ds_query_id: int
    # df_qrels. qid: int, did: int, dsqid: int (generated), dsdid: int (generated)
    def run_qrels_embs(self, docs_embs: Tensor, qs_embs: Tensor, df_docs_ids: pd.DataFrame, df_qs_ids: pd.DataFrame, df_qrels: pd.DataFrame):
        pass

    def forward(self, target_chunks: Tensor, docs_chunks: Tensor) -> Tensor:
        level_num = 1
        n_target = target_chunks.shape[0]
        inp_chunks = torch.concat((docs_chunks, target_chunks), dim=0)
        out_enc_0 = self.run_vocab_encoder(inp_chunks)
        _, out_enc_1 = self.run_encoder(level_num, out_enc_0)
        out_enc_1 = out_enc_1.unsqueeze(0)

        enc_docs_chunks, enc_target_chunks = out_enc_1[:, :-n_target], out_enc_1[:, -n_target:]
        out_dec_rank = self.decoders[0](enc_docs_chunks, enc_target_chunks)
        return out_dec_rank


class MllmRankerLevel(nn.Module):
    cfg: MllmRankerCfg
    level: int
    cfg_enc: EncoderCfg
    cfg_dec: EncoderCfg
    encoder: Encoder
    decoder: DecoderRankSimple
    # vocab_enc_cfg: Optional[VocabEncoderCfg] = None
    # vocab_encoder: Optional[VocabEncoder] = None

    def __init__(self, cfg: MllmRankerCfg, level: int):
        super().__init__()
        self.cfg = cfg.copy(deep=True)
        self.level = level
        self.cfg_enc = self.cfg.encoders[self.level]
        self.cfg_dec = self.cfg.decoders[self.level]
        self.vocab_enc_cfg = None
        self.vocab_encoder = None
        if self.level == 0:
            self.vocab_enc_cfg = self.cfg.vocab_encoder
            self.vocab_encoder = VocabEncoder(
                **self.vocab_enc_cfg.dict(),
            )
        self.encoder = Encoder(**self.cfg_enc.dict())
        self.decoder = DecoderRankSimple(self.cfg_dec.d_model)
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, -0.1, 0.1)
            pnp = p.detach().cpu().numpy()
            print(n, pnp.shape, pnp.min(), pnp.mean(), pnp.max())

    def run_vocab_encoder(self, inp: Tensor) -> Tensor:
        if self.vocab_encoder is not None:
            inp = self.vocab_encoder(inp)
        return inp

    def run_encoder(self, inp: Tensor) -> tuple[Tensor, Tensor]:
        out = self.encoder(inp)[0]
        out_seq = out_emb = out
        # if self.cfg_enc.with_emb_mat:
        #     out_seq = out_emb = out
        # else:
        #     out_seq, out_emb = out[..., :-1, :], out[..., -1, :]
        return out_seq, out_emb

    def run_qs(self, docs_chunks: Tensor, qs_chunks: Tensor, docs_off_len: list[tuple[int, int]],
               qs_off_len: list[tuple[int, int]]) -> tuple[list[Tensor], list[Tensor]]:
        n_docs, n_qs = len(docs_off_len), len(qs_off_len)
        assert n_docs == n_qs, f'# of docs ({n_docs}) != # of queries ({n_qs})'
        device = docs_chunks.device
        inp_chunks = torch.concat((docs_chunks, qs_chunks), dim=0)
        out_enc = self.run_vocab_encoder(inp_chunks)
        _, out_enc = self.run_encoder(out_enc)
        n_docs_chunks = docs_chunks.shape[0]
        docs_enc, qs_enc = out_enc[:n_docs_chunks], out_enc[n_docs_chunks:]

        qs_encs = []
        for query_off, query_len in qs_off_len:
            qs_encs.append(qs_enc[query_off:query_off + query_len])

        docs_encs = docs_enc.unsqueeze(0)
        ranks, masks = [], []
        for i_query in range(n_qs):
            query_enc = qs_encs[i_query].unsqueeze(0)
            out_rank = self.decoder(docs_encs, query_enc)
            ranks.append(out_rank)
            mask = torch.full((n_docs_chunks,), False, dtype=torch.bool, device=device)
            doc_off, doc_len = docs_off_len[i_query]
            mask[doc_off:doc_off + doc_len] = True
            masks.append(mask)

        return ranks, masks

    def run_qs_infer(self, docs_chunks: Tensor, qs_chunks: Tensor) -> Tensor:
        inp_chunks = torch.concat((docs_chunks, qs_chunks), dim=0)
        out_enc = self.run_vocab_encoder(inp_chunks)
        _, out_enc = self.run_encoder(out_enc)
        n_docs_chunks = docs_chunks.shape[0]
        docs_enc, qs_enc = out_enc[:n_docs_chunks], out_enc[n_docs_chunks:]
        out_rank = self.decoder(docs_enc.unsqueeze(0), qs_enc.unsqueeze(0))
        return out_rank

    # docs_embs: [n_batch, chunk_size, emb_size]
    # qs_embs: [n_batch * (n_queries_chunks = 1 or 2 generally, varies per query), emb_size]
    def run_qs_embs(self, docs_embs: Tensor, qs_embs: Tensor, qs_ind_len: list[tuple[int, int, int]]) -> Union[list[Tensor], Tensor]:
        # docs_enc: [n_batch, emb_size]
        _, docs_enc = self.run_encoder(docs_embs)
        # docs_enc: [1, n_batch, emb_size]
        docs_enc = docs_enc.unsqueeze(0)
        if len(qs_ind_len) == len(qs_embs):
            qs_embs = qs_embs.unsqueeze(1)
            docs_enc = docs_enc.expand((len(qs_embs), *docs_enc.shape[1:]))
            ranks = self.decoder(docs_enc, qs_embs)
        else:
            ranks = []
            for i, (qid, q_ind, q_len) in enumerate(qs_ind_len):
                # qs_embs_item: [1, n_qs, emb_size]
                qs_embs_item = qs_embs[q_ind:q_ind + q_len].unsqueeze(0)
                # out_rank: [1, n_batch]
                out_rank = self.decoder(docs_enc, qs_embs_item).squeeze(0)
                ranks.append(out_rank)
        return ranks

    def run_enc_emb(self, inp: Tensor) -> Tensor:
        out = self.run_vocab_encoder(inp)
        out = self.run_encoder(out)
        return out[1]


def test_create_mllm_ranker():
    cfg_mllm = create_mllm_ranker_cfg(n_vocab=50_000)
    print(cfg_mllm)
    mllm = MllmRanker(cfg_mllm)
    print(mllm)


if __name__ == '__main__':
    test_create_mllm_ranker()


