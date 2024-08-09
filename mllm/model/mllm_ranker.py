from typing import Union, TypeVar

import numpy as np
import torch
from torch import nn, Tensor

from mllm.exp.cfg_v1_0_0 import CfgMllmRanker, create_mllm_ranker_cfg
from mllm.model.modules import VocabEncoder, Encoder, Decoder, DecoderRankSimple


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
        out = self.run_encoder(0, out)
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

    def run_qs_3(self, docs_chunks: Tensor, qs_chunks: Tensor, docs_off_len: list[tuple[int, int]],
               qs_off_len: list[tuple[int, int]]) -> tuple[list[Tensor], list[Tensor]]:
        n_docs, n_qs = len(docs_off_len), len(qs_off_len)
        assert n_docs == n_qs, f'# of docs ({n_docs}) != # of queries ({n_qs})'
        device = docs_chunks.device
        inp_chunks = torch.concat((docs_chunks, qs_chunks), dim=0)
        out_enc = self.run_vocab_encoder(inp_chunks)
        _, out_enc = self.run_encoder(1, out_enc)
        n_docs_chunks = docs_chunks.shape[0]
        docs_enc, qs_enc = out_enc[:n_docs_chunks], out_enc[n_docs_chunks:]

        i_query = np.random.randint(n_qs)
        mask = torch.full((n_docs_chunks,), False, dtype=torch.bool, device=device)
        query_off, query_len = qs_off_len[i_query]
        doc_off, doc_len = docs_off_len[i_query]
        query_enc = qs_enc[query_off:query_off + query_len]
        mask[doc_off:doc_off + doc_len] = True
        docs_enc = docs_enc.unsqueeze(0)
        query_enc = query_enc.unsqueeze(0)
        mask = mask.unsqueeze(0)

        out_rank = self.decoders[0](docs_enc, query_enc)

        return [out_rank], [mask]

    def run_qs_2(self, docs_chunks: Tensor, qs_chunks: Tensor, docs_off_len: list[tuple[int, int]],
               qs_off_len: list[tuple[int, int]]) -> tuple[list[Tensor], list[Tensor]]:
        n_docs, n_qs = len(docs_off_len), len(qs_off_len)
        assert n_docs == n_qs, f'# of docs ({n_docs}) != # of queries ({n_qs})'
        device = docs_chunks.device
        inp_chunks = torch.concat((docs_chunks, qs_chunks), dim=0)
        out_enc = self.run_vocab_encoder(inp_chunks)
        _, out_enc = self.run_encoder(1, out_enc)
        n_docs_chunks = docs_chunks.shape[0]
        docs_enc, qs_enc = out_enc[:n_docs_chunks], out_enc[n_docs_chunks:]

        docs_encs = []
        for doc_off, doc_len in docs_off_len:
            docs_encs.append(docs_enc[doc_off:doc_off + doc_len])

        qs_encs = []
        for query_off, query_len in qs_off_len:
            qs_encs.append(qs_enc[query_off:query_off + query_len])

        ranks, masks = [], []
        for i_query in range(n_qs):
            inds_perm = torch.randperm(n_docs)
            docs_encs_perm = []
            mask = torch.full((n_docs_chunks,), False, dtype=torch.bool, device=device)
            for ind in inds_perm:
                docs_encs_perm.append(docs_encs[ind])
                if ind == i_query:
                    doc_off, doc_len = docs_off_len[ind]
                    mask[doc_off:doc_off + doc_len] = True
            docs_encs_perm = torch.concat(docs_encs_perm)
            docs_encs_perm = docs_encs_perm.unsqueeze(0)
            query_enc = qs_encs[i_query].unsqueeze(0)
            out_rank = self.decoders[0](docs_encs_perm, query_enc)
            ranks.append(out_rank)
            masks.append(mask)

        return ranks, masks

    def run_qs_1(self, docs_chunks: Tensor, qs_chunks: Tensor, docs_off_len: list[tuple[int, int]],
               qs_off_len: list[tuple[int, int]]) -> tuple[Tensor, Tensor]:
        n_docs, n_qs = len(docs_off_len), len(qs_off_len)
        assert n_docs == n_qs, f'# of docs ({n_docs}) != # of queries ({n_qs})'
        device = docs_chunks.device
        inp_chunks = torch.concat((docs_chunks, qs_chunks), dim=0)
        out_enc = self.run_vocab_encoder(inp_chunks)
        _, out_enc = self.run_encoder(1, out_enc)
        n_docs_chunks = docs_chunks.shape[0]
        doc_enc, qs_enc = out_enc[:n_docs_chunks], out_enc[n_docs_chunks:]

        masks_batch = torch.full((n_qs, len(docs_chunks)), False, dtype=torch.bool)
        ranks_batch = torch.empty((n_qs, len(docs_chunks)), dtype=torch.float32)
        for i_query in range(n_qs):
            query_off, query_len = qs_off_len[i_query]
            query_enc = qs_enc[query_off:query_off + query_len]
            docs_perm = torch.empty(doc_enc.shape, dtype=torch.float32, device=device)
            inds_perm = torch.randperm(n_docs)
            off = 0
            for doc_ind in inds_perm:
                doc_off, doc_len = docs_off_len[doc_ind]
                docs_perm[off:off + doc_len] = doc_enc[doc_off:doc_off + doc_len]
                if doc_ind == i_query:
                    masks_batch[i_query, off:off + doc_len] = True
                off += doc_len
            docs_perm, query_enc = docs_perm.unsqueeze(0), query_enc.unsqueeze(0)
            out_rank = self.decoders[0](docs_perm, query_enc)
            out_rank = out_rank.squeeze(0)
            ranks_batch[i_query] = out_rank

        return ranks_batch, masks_batch

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

    def forward_2(self, target_chunks: Tensor, docs_chunks: Tensor) -> Tensor:
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

