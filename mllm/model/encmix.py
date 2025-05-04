import sys
from typing import Optional

from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from mllm.config.model import EncmixBertCfg, EncmixOutEmbsType
from mllm.model.inference import BeamSearch
from mllm.model.mix_bert import MixBertModel

if '..' not in sys.path: sys.path.append('..')

import torch
from torch import nn


class EncmixBert(nn.Module):
    cfg: EncmixBertCfg
    tkz: PreTrainedTokenizer
    device: torch.device
    bert_model: MixBertModel
    gan_pooler: nn.Linear

    def __init__(self, cfg: EncmixBertCfg, tkz: PreTrainedTokenizer, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg
        self.tkz = tkz
        self.device = device if device is not None else torch.device('cpu')
        # if self.cfg.token_types_for_embs:
        #     type_vocab_size = 3
        # else:
        #     type_vocab_size = 2
        type_vocab_size = 2
        self.bert_model = MixBertModel.from_pretrained(
            self.cfg.pretrained_model_name, torch_dtype=torch.float32, device_map=self.device, type_vocab_size=type_vocab_size,
        )
        print(self.bert_model.config)
        # print(self.bert_model)
        assert self.tkz.pad_token_id == 0, f'pad_token_id = {self.tkz.pad_token_id}'

        if self.cfg.out_embs_type == EncmixOutEmbsType.New:
            self.out_word_embeddings = nn.Linear(self.cfg.d_model, self.tkz.vocab_size, bias=False, device=self.device)
        else:
            self.out_word_embeddings = None

        self.gan_pooler = nn.Linear(self.cfg.d_model, 1, bias=True, device=self.device)

        self._init_params()

    def _init_params(self):
        if self.out_word_embeddings is not None:
            # for n, p in self.out_word_embeddings.named_parameters():
            #     if p.dim() > 1:
            #         nn.init.xavier_uniform_(p)
            # print(self.out_word_embeddings.weight.shape, self.bert_model.embeddings.word_embeddings.weight.shape)
            self.out_word_embeddings.weight = nn.Parameter(self.bert_model.embeddings.word_embeddings.weight.clone(), requires_grad=True)

    # chunk_toks: [n_chunks, seq_len]
    # plain_toks: [n_plain_toks]
    # target_toks: [n_target_toks]
    # out_logits: [n_target_toks]
    def run_chunks_plain_seq(self, chunk_toks: torch.Tensor, target_toks: torch.Tensor, plain_toks: Optional[torch.Tensor] = None) -> torch.Tensor:
        n_chunks = chunk_toks.shape[0]
        chunk_toks_mask = chunk_toks != self.tkz.pad_token_id
        # [n_chunks, seq_len] -> [n_chunks, seq_len, d_model]
        token_type_ids = None
        if self.cfg.token_types_for_embs:
            token_type_ids = torch.zeros(chunk_toks.shape, dtype=torch.int64, device=self.device)
        chunks_out: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
            input_ids=chunk_toks, attention_mask=chunk_toks_mask, token_type_ids=token_type_ids,
        )
        # [n_chunks, seq_len, d_model] -> [n_chunks, d_model]
        chunks_emb = chunks_out.last_hidden_state[: ,0]
        # [1, 1, d_model]
        cls_wemb = self.bert_model.embeddings.word_embeddings(torch.tensor([[self.tkz.cls_token_id]], device=self.device))
        # [n_chunks + 1, d_model]
        chunks_emb = torch.concatenate([cls_wemb[0], chunks_emb], dim=0)
        n_chunks += 1

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
            # Remove first CLS token
            if plain_toks[0] == self.tkz.cls_token_id:
                plain_toks = plain_toks[1:]
            n_plain_toks = len(plain_toks)
            # [n_plain_toks] -> [n_target_toks, n_plain_toks]
            plain_toks_inp = plain_toks.repeat(n_target_toks, 1)
            # [n_target_toks, n_plain_toks], [n_target_toks, n_target_toks] -> [n_target_toks, n_plain_toks + n_target_toks]
            toks_inp = torch.concatenate([plain_toks_inp, target_toks_inp], dim=1)

        toks_inp_mask = toks_inp != self.tkz.pad_token_id
        chunks_mask = torch.ones((n_target_toks, n_chunks), dtype=torch.bool, device=self.device)
        # [n_target_toks, n_chunks], [n_target_toks, n_plain_toks + n_target_toks] -> [n_target_toks, n_chunks + n_plain_toks + n_target_toks]
        # seq_len_out = n_chunks + n_plain_toks + n_target_toks
        # [n_target_toks, seq_len_out]
        inp_mask = torch.concatenate([chunks_mask, toks_inp_mask], dim=1)
        if self.cfg.token_types_for_embs:
            token_type_ids = torch.zeros(inp_mask.shape, dtype=torch.int64, device=self.device)
            token_type_ids[:, 1:n_chunks] = 1
        mix_out: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
            inputs_starting_embeds=chunks_emb, input_ids=toks_inp, attention_mask=inp_mask, token_type_ids=token_type_ids,
        )

        # [n_target_toks, seq_len_out, d_model]
        lhs = mix_out.last_hidden_state
        # [n_target_toks, n_target_toks, d_model]
        out_logits = lhs[:, n_chunks + n_plain_toks:][target_mask]
        # out_logits = lhs[:, 1:n_target_toks + 1][target_mask]
        if self.cfg.out_embs_type == EncmixOutEmbsType.Non:
            pass
        elif self.cfg.out_embs_type == EncmixOutEmbsType.Inp:
            # [d_model, vocab_size]
            wemb_weights = self.bert_model.embeddings.word_embeddings.weight
            # [n_target_toks, vocab_size]
            out_logits = out_logits @ wemb_weights.T
        else:
            # [n_target_toks, vocab_size]
            out_logits = self.out_word_embeddings(out_logits)

        return out_logits

    # chunk_toks: [n_chunks, seq_len]
    # plain_toks: [n_plain_toks]
    # target_toks: [n_target_toks]
    # out_toks: [<=max_out_toks]
    def predict(self, chunk_toks: torch.Tensor, plain_toks: Optional[torch.Tensor] = None, max_out_toks: int = 20) -> torch.Tensor:
        n_chunks = chunk_toks.shape[0]
        chunk_toks_mask = chunk_toks != self.tkz.pad_token_id
        token_type_ids = None
        if self.cfg.token_types_for_embs:
            token_type_ids = torch.zeros(chunk_toks.shape, dtype=torch.int64, device=self.device)
        chunks_out: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
            input_ids=chunk_toks, attention_mask=chunk_toks_mask, token_type_ids=token_type_ids,
        )
        # [n_chunks, seq_len, d_model] -> [n_chunks, d_model]
        chunks_emb = chunks_out.last_hidden_state[: ,0]
        # [1, 1, d_model]
        cls_wemb = self.bert_model.embeddings.word_embeddings(torch.tensor([[self.tkz.cls_token_id]], device=self.device))
        # [n_chunks + 1, d_model]
        chunks_emb = torch.concatenate([cls_wemb[0], chunks_emb], dim=0)
        n_chunks += 1

        # [n_chunks, d_model] -> [1, n_chunks, d_model]
        chunks_emb = chunks_emb.unsqueeze(0)
        mask_toks_single = torch.tensor([self.tkz.mask_token_id], dtype=chunk_toks.dtype, device=self.device)
        target_toks = mask_toks_single
        while True:
            if plain_toks is None:
                n_plain_toks = 0
                toks_inp = target_toks
            else:
                n_plain_toks = len(plain_toks)
                # [n_plain_toks], [n_target_toks] -> [n_plain_toks + n_target_toks]
                toks_inp = torch.concatenate([plain_toks, target_toks])
            # [n_plain_toks + n_target_toks] -> [1, n_plain_toks + n_target_toks]
            toks_inp = toks_inp.unsqueeze(0)

            toks_inp_mask = toks_inp != self.tkz.pad_token_id
            chunks_mask = torch.ones((1, n_chunks), dtype=torch.bool, device=self.device)
            # [1, n_chunks], [1, n_plain_toks + n_target_toks] -> [1, n_chunks + n_plain_toks + n_target_toks]
            # seq_len_out = n_chunks + n_plain_toks + n_target_toks
            # [1, seq_len_out]
            inp_mask = torch.concatenate([chunks_mask, toks_inp_mask], dim=1)
            if self.cfg.token_types_for_embs:
                token_type_ids = torch.zeros(inp_mask.shape, dtype=torch.int64, device=self.device)
                token_type_ids[:, 1:n_chunks] = 1
            mix_out: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
                inputs_starting_embeds=chunks_emb, input_ids=toks_inp, attention_mask=inp_mask, token_type_ids=token_type_ids,
            )

            # [1, seq_len_out, d_model]
            lhs = mix_out.last_hidden_state
            # [1, d_model]
            out_logits = lhs[:, -1]
            # out_logits = lhs[:, 1:n_target_toks + 1][target_mask]
            if self.cfg.out_embs_type == EncmixOutEmbsType.Non:
                raise Exception(f'Out embeddings type {self.cfg.out_embs_type} is not supported')
            if self.cfg.out_embs_type == EncmixOutEmbsType.Inp:
                # [d_model, vocab_size]
                wemb_weights = self.bert_model.embeddings.word_embeddings.weight
                # [1, vocab_size]
                out_logits = out_logits @ wemb_weights.T
            else:
                # [1, vocab_size]
                out_logits = self.out_word_embeddings(out_logits)

            # [1, vocab_size]
            probs_pred = torch.softmax(out_logits, dim=-1)
            # [1]
            toks_pred = torch.argmax(probs_pred, dim=-1)

            target_toks = torch.concatenate([target_toks[:-1], toks_pred, mask_toks_single])
            # print(toks_pred.shape, target_toks.shape)
            if toks_pred.squeeze().item() == self.tkz.sep_token_id or len(target_toks) == max_out_toks:
                target_toks = target_toks[:-1]
                break

        return target_toks

    # chunk_toks: [n_chunks, seq_len]
    # plain_toks: [n_plain_toks]
    # target_toks: [n_target_toks]
    # out_toks: [<=max_out_toks]
    def predict_beam(self, chunk_toks: torch.Tensor, plain_toks: Optional[torch.Tensor] = None, max_out_toks: int = 20,
                     num_beams: int = 5, temperature: float = 1,) -> list[int]:
        beam_search = BeamSearch(
            num_beams=num_beams, max_len=max_out_toks, temperature=temperature, next_token_id=self.tkz.mask_token_id,
            last_token_id=self.tkz.sep_token_id, device=self.device,
        )

        n_chunks = chunk_toks.shape[0]
        chunk_toks_mask = chunk_toks != self.tkz.pad_token_id
        token_type_ids = None
        if self.cfg.token_types_for_embs:
            token_type_ids = torch.zeros(chunk_toks.shape, dtype=torch.int64, device=self.device)
        chunks_out: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
            input_ids=chunk_toks, attention_mask=chunk_toks_mask, token_type_ids=token_type_ids,
        )
        # [n_chunks, seq_len, d_model] -> [n_chunks, d_model]
        chunks_emb = chunks_out.last_hidden_state[: ,0]
        # [1, 1, d_model]
        cls_wemb = self.bert_model.embeddings.word_embeddings(torch.tensor([[self.tkz.cls_token_id]], device=self.device))
        # [n_chunks + 1, d_model]
        chunks_emb = torch.concatenate([cls_wemb[0], chunks_emb], dim=0)
        n_chunks += 1

        # toks_inp: [n_active_beams, beam_seq_len] -> [n_active_beams, vocab_size]
        def run_inference(beam_seq_batch: torch.Tensor) -> torch.Tensor:
            n_active_beams = beam_seq_batch.shape[0]
            # [n_active_beams, beam_seq_len]
            toks_inp = beam_seq_batch
            if plain_toks is not None:
                plain_toks_inp = plain_toks.repeat((n_active_beams, 1))
                # [n_active_beams, n_plain_toks + beam_seq_len]
                # print(plain_toks_inp.shape, beam_seq_batch.shape)
                toks_inp = torch.concatenate([plain_toks_inp, beam_seq_batch], dim=-1)

            # [n_active_beams, n_chunks, d_model]
            chunks_emb_inp = chunks_emb.repeat((n_active_beams, 1, 1))
            # chunks_emb_inp = chunks_emb_inp[:, :1]

            # [n_active_beams, n_plain_toks + beam_seq_len]
            toks_inp_mask = toks_inp != self.tkz.pad_token_id
            # [n_active_beams, n_chunks]
            chunks_mask = torch.ones((n_active_beams, n_chunks), dtype=torch.bool, device=self.device)
            # [n_active_beams, n_chunks + n_plain_toks + beam_seq_len]
            inp_mask = torch.concatenate([chunks_mask, toks_inp_mask], dim=1)
            token_type_ids = None
            if self.cfg.token_types_for_embs:
                token_type_ids = torch.zeros(inp_mask.shape, dtype=torch.int64, device=self.device)
                token_type_ids[:, 1:n_chunks] = 1
            # print(f'chunks_emb_inp: {chunks_emb_inp.shape}. toks_inp: {toks_inp.shape}, inp_mask: {inp_mask.shape} all: {inp_mask.all()}. token_type_ids: {token_type_ids.shape}')
            mix_out: BaseModelOutputWithPoolingAndCrossAttentions = self.bert_model(
                inputs_starting_embeds=chunks_emb_inp, input_ids=toks_inp, attention_mask=inp_mask, token_type_ids=token_type_ids,
            )

            # [n_active_beams, n_chunks + n_plain_toks + beam_seq_len, d_model]
            lhs = mix_out.last_hidden_state
            # [n_active_beams, d_model]
            out_logits = lhs[:, -1]
            if self.cfg.out_embs_type == EncmixOutEmbsType.Non:
                raise Exception(f'Out embeddings type {self.cfg.out_embs_type} is not supported')
            if self.cfg.out_embs_type == EncmixOutEmbsType.Inp:
                # [d_model, vocab_size]
                wemb_weights = self.bert_model.embeddings.word_embeddings.weight
                # [n_active_beams, vocab_size]
                out_logits = out_logits @ wemb_weights.T
            else:
                # [n_active_beams, vocab_size]
                out_logits = self.out_word_embeddings(out_logits)

            return out_logits

        beams = beam_search.run(run_inference)
        for beam in beams:
            print(self.tkz.decode(beam.tokens_cur))

        return beams[0].tokens_cur


class EncmixBertGan(nn.Module):
    cfg: EncmixBertCfg
    tkz: PreTrainedTokenizer
    device: torch.device
    gen_model: MixBertModel
    dis_model: MixBertModel
    dis_pooler: nn.Linear

    def __init__(self, cfg: EncmixBertCfg, tkz: PreTrainedTokenizer, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg
        self.tkz = tkz
        assert self.tkz.pad_token_id == 0, f'pad_token_id = {self.tkz.pad_token_id}'
        self.device = device if device is not None else torch.device('cpu')
        self.gen_model = MixBertModel.from_pretrained(
            self.cfg.pretrained_model_name, torch_dtype=torch.float32, device_map=self.device,
        )
        self.dis_model = MixBertModel.from_pretrained(
            self.cfg.pretrained_model_name, torch_dtype=torch.float32, device_map=self.device,
        )

        assert self.cfg.out_embs_type == EncmixOutEmbsType.Inp
        self.out_word_embeddings = None

        # Tie weights
        self.dis_model.embeddings = self.gen_model.embeddings

        self.dis_pooler = nn.Linear(self.cfg.d_model, 1, bias=True, device=self.device)

        self._init_params()

    def _init_params(self):
        if self.out_word_embeddings is not None:
            # for n, p in self.out_word_embeddings.named_parameters():
            #     if p.dim() > 1:
            #         nn.init.xavier_uniform_(p)
            # print(self.out_word_embeddings.weight.shape, self.bert_model.embeddings.word_embeddings.weight.shape)
            self.out_word_embeddings.weight = nn.Parameter(self.bert_model.embeddings.word_embeddings.weight.clone(), requires_grad=True)

    def _tkz_inp(self, s: str, max_len: Optional[int] = None, strip: bool = True) -> torch.Tensor:
        toks = self.tkz(s)['input_ids']
        assert toks[0] == self.tkz.cls_token_id and toks[-1] == self.tkz.sep_token_id, f'toks = {toks}. cls_token_id = {self.tkz.cls_token_id}. sep_token_id = {self.tkz.sep_token_id}'
        if max_len is not None and len(toks) > max_len:
            toks = toks[:max_len - 1] + [toks[-1]]
        if strip:
            toks = toks[1:-1]
        toks_t = torch.tensor(toks, device=self.device)
        toks_t = toks_t.unsqueeze(0)
        return toks_t

    def get_gen_parameters(self) -> list[nn.parameter.Parameter]:
        res = []
        for name, param in self.gen_model.named_parameters():
            # print(name)
            res.append(param)
        return res

    def get_dis_parameters(self) -> list[nn.parameter.Parameter]:
        res = []
        for name, param in self.dis_model.named_parameters():
            # print(name)
            if name.startswith('embeddings.'):
                continue
            res.append(param)
        for name, param in self.dis_pooler.named_parameters():
            res.append(param)
        return res

    def run_qna_gan(self, context: str, question: str, answer: str, is_gen: bool) -> torch.Tensor:
        # [1, n_ctx] = [CLS, TOK*, SEP]
        c_toks_t = self._tkz_inp(context, max_len=self.cfg.inp_len, strip=False)
        assert c_toks_t[0, 0] == self.tkz.cls_token_id and c_toks_t[0, -1] == self.tkz.sep_token_id, f'{c_toks_t}. cls_tok_id = {self.tkz.cls_token_id}. sep_tok_id = {self.tkz.sep_token_id}'

        token_type_ids = None
        if self.cfg.token_types_for_embs:
            token_type_ids = torch.zeros((1, c_toks_t.shape[1]), dtype=torch.int64, device=self.device)
        c_out: BaseModelOutputWithPoolingAndCrossAttentions = self.gen_model(
            input_ids=c_toks_t, attention_mask=c_toks_t != self.tkz.pad_token_id, token_type_ids=token_type_ids,
        )
        # [1, d_model]
        c_emb = c_out.last_hidden_state[:, 0]
        # [1, 1, d_model]
        c_emb = c_emb.unsqueeze(0)

        q_str = f'Question: {question}'
        acap_str = ' Answer: '
        qacap_str = f'{q_str} {acap_str}'
        # [1, n_qa] = [TOK*]
        qacap_toks_t = self._tkz_inp(qacap_str)
        assert qacap_toks_t[0, 0] != self.tkz.cls_token_id and qacap_toks_t[0, -1] != self.tkz.sep_token_id, f'{qacap_toks_t}. cls_tok_id = {self.tkz.cls_token_id}. sep_tok_id = {self.tkz.sep_token_id}'
        # [1, n_qa, d_model]
        qacap_wemb = self.gen_model.embeddings.word_embeddings(qacap_toks_t)

        # max_out_toks = 20
        # [1, n_ans] = [TOK*]
        a_toks_t = self._tkz_inp(answer)
        # if a_toks_t.shape[1] > max_out_toks:
        #     a_toks_t = a_toks_t[:, :max_out_toks]
        # assert a_toks_t[0, 0] != self.tkz.cls_token_id and a_toks_t[0, -1] != self.tkz.sep_token_id, f'{a_toks_t}. cls_tok_id = {self.tkz.cls_token_id}. sep_tok_id = {self.tkz.sep_token_id}'
        n_ans = a_toks_t.shape[1]
        max_out_toks = n_ans

        # [1, max_out_toks]
        amask_toks_t = torch.ones((1, max_out_toks), dtype=torch.int64, device=self.device) * self.tkz.mask_token_id

        # [1]
        cls_toks_t = torch.tensor([[self.tkz.cls_token_id]], device=self.device)
        # [1, 1, d_model]
        cls_wemb = self.gen_model.embeddings.word_embeddings(cls_toks_t)

        # [1]
        sep_toks_t = torch.tensor([[self.tkz.sep_token_id]], device=self.device)
        # [1, 1, d_model]
        sep_wemb = self.gen_model.embeddings.word_embeddings(sep_toks_t)

        # [1, 2, d_model]
        cc_emb = torch.concatenate([cls_wemb, c_emb], dim=1)

        # [1, n_qa + max_out_toks + 1]
        qam_toks_t = torch.concatenate([qacap_toks_t, amask_toks_t, sep_toks_t], dim=-1)

        if self.cfg.token_types_for_embs:
            token_type_ids = torch.zeros((1, 2 + qam_toks_t.shape[1]), dtype=torch.int64, device=self.device)
            token_type_ids[:, 1] = 1
        qam_out: BaseModelOutputWithPoolingAndCrossAttentions = self.gen_model(
            inputs_starting_embeds=cc_emb, input_ids=qam_toks_t, token_type_ids=token_type_ids,
        )
        # [1, n_ctx + n_qa + max_out_toks]
        qam_lhs = qam_out.last_hidden_state
        # [1, max_out_toks]
        a_emb_pred = qam_lhs[:, -max_out_toks - 1:-1]

        # [1, n_ctx, d_model]
        c_wemb = self.gen_model.embeddings.word_embeddings(c_toks_t)

        # [1, n_ans, d_model]
        a_wemb = self.gen_model.embeddings.word_embeddings(a_toks_t)

        tgt1, a1_emb = 0, a_emb_pred
        # [1, 1 + 1 + n_qacap + n_ans]
        inp1_emb = torch.concatenate([cls_wemb, c_wemb[:, 1:-1], qacap_wemb, a1_emb], dim=1)

        if self.cfg.token_types_for_embs:
            token_type_ids = torch.zeros((1, inp1_emb.shape[1] + 1), dtype=torch.int64, device=self.device)
        gan_out1: BaseModelOutputWithPoolingAndCrossAttentions = self.dis_model(
            inputs_starting_embeds=inp1_emb, input_ids=sep_toks_t, token_type_ids=token_type_ids,
        )
        # [1, d_model]
        dis_pooler_out1 = gan_out1.last_hidden_state[:, 0]
        # [1, 1]
        dis_logits1 = self.dis_pooler(dis_pooler_out1)
        dis_logits = [dis_logits1]
        targets = [tgt1]

        if not is_gen:
            tgt2, a2_emb = 1, a_wemb
            # [1, 1 + 1 + n_qacap + n_ans]
            inp2_emb = torch.concatenate([cls_wemb, c_wemb[:, 1:-1], qacap_wemb, a2_emb], dim=1)
            if self.cfg.token_types_for_embs:
                token_type_ids = torch.zeros((1, inp2_emb.shape[1] + 1), dtype=torch.int64, device=self.device)
            gan_out2: BaseModelOutputWithPoolingAndCrossAttentions = self.dis_model(
                inputs_starting_embeds=inp2_emb, input_ids=sep_toks_t, token_type_ids=token_type_ids,
            )
            # [1, d_model]
            dis_pooler_out2 = gan_out2.last_hidden_state[:, 0]
            # [1, 1]
            dis_logits2 = self.dis_pooler(dis_pooler_out2)
            dis_logits.append(dis_logits2)
            targets.append(tgt2)

        # [2, 1]
        dis_logits = torch.concatenate(dis_logits)
        # [2]
        targets = torch.tensor(targets, device=self.device)
        # [2, 1]
        targets = targets.unsqueeze(-1)

        if is_gen:
            prob = torch.sigmoid(dis_logits.squeeze())
            loss_gen = -torch.log(prob)
            return loss_gen

        probs = torch.sigmoid(dis_logits)
        loss_dis = -targets * torch.log(probs) - (1 - targets) * torch.log(1 - probs)
        loss_dis = torch.mean(loss_dis)
        return loss_dis

    def predict(self, context: str, question: str, max_out_toks: int = 20) -> tuple[torch.Tensor, str]:
        # [1, n_ctx] = [CLS, TOK*, SEP]
        c_toks_t = self._tkz_inp(context, max_len=self.cfg.inp_len, strip=False)
        assert c_toks_t[0, 0] == self.tkz.cls_token_id and c_toks_t[0, -1] == self.tkz.sep_token_id, f'{c_toks_t}. cls_tok_id = {self.tkz.cls_token_id}. sep_tok_id = {self.tkz.sep_token_id}'
        c_out: BaseModelOutputWithPoolingAndCrossAttentions = self.gen_model(
            input_ids=c_toks_t, attention_mask = c_toks_t != self.tkz.pad_token_id,
        )
        # [1, 1, d_model]
        c_emb = c_out.last_hidden_state[:, :1]

        c_logits = c_emb @ self.gen_model.embeddings.word_embeddings.weight.T
        c_probs = torch.softmax(c_logits, dim=-1)
        c_toks = torch.argmax(c_probs.squeeze(), dim=-1)
        c_str = self.tkz.decode(c_toks)
        print('Ctx emb:', c_str)

        q_str = f'Question: {question}'
        acap_str = ' Answer: '
        qacap_str = f'{q_str} {acap_str}'
        # [1, n_qa] = [TOK*]
        qacap_toks_t = self._tkz_inp(qacap_str)
        assert qacap_toks_t[0, 0] != self.tkz.cls_token_id and qacap_toks_t[0, -1] != self.tkz.sep_token_id, f'{qacap_toks_t}. cls_tok_id = {self.tkz.cls_token_id}. sep_tok_id = {self.tkz.sep_token_id}'

        # [1, max_out_toks]
        amask_toks_t = torch.ones((1, max_out_toks), dtype=torch.int64, device=self.device) * self.tkz.mask_token_id

        cls_toks_t = torch.tensor([[self.tkz.cls_token_id]], device=self.device)
        # [1, 1, d_model]
        cls_wemb = self.gen_model.embeddings.word_embeddings(cls_toks_t)

        sep_toks_t = torch.tensor([[self.tkz.sep_token_id]], device=self.device)
        # [1, 1, d_model]
        sep_wemb = self.gen_model.embeddings.word_embeddings(sep_toks_t)

        # [1, 2, d_model]
        cc_emb = torch.concatenate([cls_wemb, c_emb], dim=1)

        # [1, n_qa + max_out_toks + 1]
        qam_toks_t = torch.concatenate([qacap_toks_t, amask_toks_t, sep_toks_t], dim=-1)
        print(cc_emb.shape)
        print(self.tkz.decode(qam_toks_t[0]))
        qam_out: BaseModelOutputWithPoolingAndCrossAttentions = self.gen_model(
            inputs_starting_embeds=cc_emb, input_ids=qam_toks_t, attention_mask=qam_toks_t != self.tkz.pad_token_id,
        )
        # [1, 1 + 1 + n_qa + max_out_toks + 1]
        qam_lhs = qam_out.last_hidden_state
        # [1, max_out_toks]
        ans_embs = qam_lhs[:, -max_out_toks - 1:-1]

        # [1, max_out_toks, vocab_size]
        ans_logits = ans_embs @ self.gen_model.embeddings.word_embeddings.weight.T
        # [1, max_out_toks, vocab_size]
        ans_probs = torch.softmax(ans_logits, dim=-1)
        # [1, max_out_toks]
        ans_toks = torch.argmax(ans_probs.squeeze(), dim=-1)
        ans_str = self.tkz.decode(ans_toks)
        return ans_toks, ans_str


def qna_gan_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    assert logits.ndim == target.ndim
    probs = torch.sigmoid(logits)
    loss_gen = -(1 - target) * torch.log(probs)
    loss_dis = -target * torch.log(probs) - (1 - target) * torch.log(1 - probs)
    loss = (loss_gen + loss_dis) / 2
    loss_gen, loss_dis, loss = torch.mean(loss_gen), torch.mean(loss_dis), torch.mean(loss)
    return loss, loss_gen, loss_dis


class EncmixBertSep(nn.Module):
    cfg: EncmixBertCfg
    tkz: PreTrainedTokenizer
    device: torch.device
    enc_model: MixBertModel
    dec_model: MixBertModel

    def __init__(self, cfg: EncmixBertCfg, tkz: PreTrainedTokenizer, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg
        self.tkz = tkz
        self.device = device if device is not None else torch.device('cpu')
        type_vocab_size = 2
        self.enc_model = MixBertModel.from_pretrained(
            self.cfg.pretrained_model_name, torch_dtype=torch.float32, device_map=self.device, type_vocab_size=type_vocab_size,
        )
        self.dec_model = MixBertModel.from_pretrained(
            self.cfg.pretrained_model_name, torch_dtype=torch.float32, device_map=self.device, type_vocab_size=type_vocab_size,
        )
        print(self.dec_model.config)
        # print(self.bert_model)
        assert self.tkz.pad_token_id == 0, f'pad_token_id = {self.tkz.pad_token_id}'

        if self.cfg.out_embs_type == EncmixOutEmbsType.New:
            self.out_word_embeddings = nn.Linear(self.cfg.d_model, self.tkz.vocab_size, bias=False, device=self.device)
        else:
            self.out_word_embeddings = None

        self._init_params()

    def _init_params(self):
        if self.out_word_embeddings is not None:
            # for n, p in self.out_word_embeddings.named_parameters():
            #     if p.dim() > 1:
            #         nn.init.xavier_uniform_(p)
            # print(self.out_word_embeddings.weight.shape, self.bert_model.embeddings.word_embeddings.weight.shape)
            self.out_word_embeddings.weight = nn.Parameter(self.dec_model.embeddings.word_embeddings.weight.clone(), requires_grad=True)

    def _get_token_type_ids(self, shape: tuple[int], target_range: Optional[range] = None) -> Optional[torch.Tensor]:
        token_type_ids = None
        if self.cfg.token_types_for_embs:
            token_type_ids = torch.zeros(shape, dtype=torch.int64, device=self.device)
            if target_range is not None:
                token_type_ids[:, target_range] = 1
        return token_type_ids

    # chunk_toks: [n_chunks, seq_len]
    # plain_toks: [n_plain_toks]
    # target_toks: [n_target_toks]
    # out_logits: [n_target_toks]
    def run_chunks_plain_seq(self, chunk_toks: torch.Tensor, target_toks: torch.Tensor, plain_toks: Optional[torch.Tensor] = None) -> torch.Tensor:
        n_chunks = chunk_toks.shape[0]
        # [n_chunks, seq_len]
        chunk_toks_mask = chunk_toks != self.tkz.pad_token_id
        # [n_chunks, seq_len]
        token_type_ids = self._get_token_type_ids(chunk_toks.shape)
        chunks_out: BaseModelOutputWithPoolingAndCrossAttentions = self.enc_model(
            input_ids=chunk_toks, attention_mask=chunk_toks_mask, token_type_ids=token_type_ids,
        )
        # [n_chunks, seq_len, d_model]
        chunks_lhs = chunks_out.last_hidden_state
        # [n_chunks, d_model]
        chunks_emb = chunks_lhs[:, 0]
        # [1, 1, d_model]
        cls_wemb = self.dec_model.embeddings.word_embeddings(torch.tensor([[self.tkz.cls_token_id]], device=self.device))
        # [n_chunks + 1, d_model]
        chunks_emb = torch.concatenate([cls_wemb[0], chunks_emb], dim=0)
        n_chunks += 1

        n_target_toks = len(target_toks)

        # [n_target_toks] -> [n_target_toks, n_target_toks]
        target_toks_inp = target_toks.repeat(n_target_toks, 1)
        target_toks_inp = torch.tril(target_toks_inp)
        target_mask = torch.eye(n_target_toks, dtype=torch.bool, device=self.device)
        target_toks_inp[target_mask] = self.tkz.mask_token_id

        # [n_chunks, d_model] -> [n_target_toks, n_chunks, d_model]
        chunks_emb = chunks_emb.repeat(n_target_toks, 1, 1)

        if plain_toks is None:
            n_plain_toks = 0
            toks_inp = target_toks_inp
        else:
            # Remove first CLS token
            if plain_toks[0] == self.tkz.cls_token_id:
                plain_toks = plain_toks[1:]
            n_plain_toks = len(plain_toks)
            # [n_plain_toks] -> [n_target_toks, n_plain_toks]
            plain_toks_inp = plain_toks.repeat(n_target_toks, 1)
            # [n_target_toks, n_plain_toks], [n_target_toks, n_target_toks] -> [n_target_toks, n_plain_toks + n_target_toks]
            toks_inp = torch.concatenate([plain_toks_inp, target_toks_inp], dim=1)

        toks_inp_mask = toks_inp != self.tkz.pad_token_id
        chunks_mask = torch.ones((n_target_toks, n_chunks), dtype=torch.bool, device=self.device)
        # [n_target_toks, n_chunks], [n_target_toks, n_plain_toks + n_target_toks] -> [n_target_toks, n_chunks + n_plain_toks + n_target_toks]
        # seq_len_out = n_chunks + n_plain_toks + n_target_toks
        # [n_target_toks, seq_len_out]
        inp_mask = torch.concatenate([chunks_mask, toks_inp_mask], dim=1)
        token_type_ids = self._get_token_type_ids(inp_mask.shape, range(1, n_chunks))
        mix_out: BaseModelOutputWithPoolingAndCrossAttentions = self.dec_model(
            inputs_starting_embeds=chunks_emb, input_ids=toks_inp, attention_mask=inp_mask, token_type_ids=token_type_ids,
        )

        # [n_target_toks, seq_len_out, d_model]
        lhs = mix_out.last_hidden_state
        # [n_target_toks, n_target_toks, d_model]
        out_logits = lhs[:, n_chunks + n_plain_toks:][target_mask]
        # out_logits = lhs[:, 1:n_target_toks + 1][target_mask]
        if self.cfg.out_embs_type == EncmixOutEmbsType.Non:
            pass
        elif self.cfg.out_embs_type == EncmixOutEmbsType.Inp:
            # [d_model, vocab_size]
            wemb_weights = self.dec_model.embeddings.word_embeddings.weight
            # [n_target_toks, vocab_size]
            out_logits = out_logits @ wemb_weights.T
        else:
            # [n_target_toks, vocab_size]
            out_logits = self.out_word_embeddings(out_logits)

        return out_logits

    # chunk_toks: [n_chunks, seq_len]
    # plain_toks: [n_plain_toks]
    # target_toks: [n_target_toks]
    # out_toks: [<=max_out_toks]
    def predict(self, chunk_toks: torch.Tensor, plain_toks: Optional[torch.Tensor] = None, max_out_toks: int = 20) -> torch.Tensor:
        n_chunks = chunk_toks.shape[0]
        chunk_toks_mask = chunk_toks != self.tkz.pad_token_id
        token_type_ids = None
        if self.cfg.token_types_for_embs:
            token_type_ids = torch.zeros(chunk_toks.shape, dtype=torch.int64, device=self.device)
        chunks_out: BaseModelOutputWithPoolingAndCrossAttentions = self.dec_model(
            input_ids=chunk_toks, attention_mask=chunk_toks_mask, token_type_ids=token_type_ids,
        )
        # [n_chunks, seq_len, d_model] -> [n_chunks, d_model]
        chunks_emb = chunks_out.last_hidden_state[: ,0]
        # [1, 1, d_model]
        cls_wemb = self.dec_model.embeddings.word_embeddings(torch.tensor([[self.tkz.cls_token_id]], device=self.device))
        # [n_chunks + 1, d_model]
        chunks_emb = torch.concatenate([cls_wemb[0], chunks_emb], dim=0)
        n_chunks += 1

        # [n_chunks, d_model] -> [1, n_chunks, d_model]
        chunks_emb = chunks_emb.unsqueeze(0)
        mask_toks_single = torch.tensor([self.tkz.mask_token_id], dtype=chunk_toks.dtype, device=self.device)
        target_toks = mask_toks_single
        while True:
            if plain_toks is None:
                n_plain_toks = 0
                toks_inp = target_toks
            else:
                n_plain_toks = len(plain_toks)
                # [n_plain_toks], [n_target_toks] -> [n_plain_toks + n_target_toks]
                toks_inp = torch.concatenate([plain_toks, target_toks])
            # [n_plain_toks + n_target_toks] -> [1, n_plain_toks + n_target_toks]
            toks_inp = toks_inp.unsqueeze(0)

            toks_inp_mask = toks_inp != self.tkz.pad_token_id
            chunks_mask = torch.ones((1, n_chunks), dtype=torch.bool, device=self.device)
            # [1, n_chunks], [1, n_plain_toks + n_target_toks] -> [1, n_chunks + n_plain_toks + n_target_toks]
            # seq_len_out = n_chunks + n_plain_toks + n_target_toks
            # [1, seq_len_out]
            inp_mask = torch.concatenate([chunks_mask, toks_inp_mask], dim=1)
            if self.cfg.token_types_for_embs:
                token_type_ids = torch.zeros(inp_mask.shape, dtype=torch.int64, device=self.device)
                token_type_ids[:, 1:n_chunks] = 1
            mix_out: BaseModelOutputWithPoolingAndCrossAttentions = self.dec_model(
                inputs_starting_embeds=chunks_emb, input_ids=toks_inp, attention_mask=inp_mask, token_type_ids=token_type_ids,
            )

            # [1, seq_len_out, d_model]
            lhs = mix_out.last_hidden_state
            # [1, d_model]
            out_logits = lhs[:, -1]
            # out_logits = lhs[:, 1:n_target_toks + 1][target_mask]
            if self.cfg.out_embs_type == EncmixOutEmbsType.Non:
                raise Exception(f'Out embeddings type {self.cfg.out_embs_type} is not supported')
            if self.cfg.out_embs_type == EncmixOutEmbsType.Inp:
                # [d_model, vocab_size]
                wemb_weights = self.dec_model.embeddings.word_embeddings.weight
                # [1, vocab_size]
                out_logits = out_logits @ wemb_weights.T
            else:
                # [1, vocab_size]
                out_logits = self.out_word_embeddings(out_logits)

            # [1, vocab_size]
            probs_pred = torch.softmax(out_logits, dim=-1)
            # [1]
            toks_pred = torch.argmax(probs_pred, dim=-1)

            target_toks = torch.concatenate([target_toks[:-1], toks_pred, mask_toks_single])
            # print(toks_pred.shape, target_toks.shape)
            if toks_pred.squeeze().item() == self.tkz.sep_token_id or len(target_toks) == max_out_toks:
                target_toks = target_toks[:-1]
                break

        return target_toks

    # chunk_toks: [n_chunks, seq_len]
    # plain_toks: [n_plain_toks]
    # target_toks: [n_target_toks]
    # out_toks: [<=max_out_toks]
    def predict_beam(self, chunk_toks: torch.Tensor, plain_toks: Optional[torch.Tensor] = None, max_out_toks: int = 20,
                     num_beams: int = 5, temperature: float = 1,) -> list[int]:
        beam_search = BeamSearch(
            num_beams=num_beams, max_len=max_out_toks, temperature=temperature, next_token_id=self.tkz.mask_token_id,
            last_token_id=self.tkz.sep_token_id, device=self.device,
        )

        n_chunks = chunk_toks.shape[0]
        chunk_toks_mask = chunk_toks != self.tkz.pad_token_id
        token_type_ids = None
        if self.cfg.token_types_for_embs:
            token_type_ids = torch.zeros(chunk_toks.shape, dtype=torch.int64, device=self.device)
        chunks_out: BaseModelOutputWithPoolingAndCrossAttentions = self.dec_model(
            input_ids=chunk_toks, attention_mask=chunk_toks_mask, token_type_ids=token_type_ids,
        )
        # [n_chunks, seq_len, d_model] -> [n_chunks, d_model]
        chunks_emb = chunks_out.last_hidden_state[: ,0]
        # [1, 1, d_model]
        cls_wemb = self.dec_model.embeddings.word_embeddings(torch.tensor([[self.tkz.cls_token_id]], device=self.device))
        # [n_chunks + 1, d_model]
        chunks_emb = torch.concatenate([cls_wemb[0], chunks_emb], dim=0)
        n_chunks += 1

        # toks_inp: [n_active_beams, beam_seq_len] -> [n_active_beams, vocab_size]
        def run_inference(beam_seq_batch: torch.Tensor) -> torch.Tensor:
            n_active_beams = beam_seq_batch.shape[0]
            # [n_active_beams, beam_seq_len]
            toks_inp = beam_seq_batch
            if plain_toks is not None:
                plain_toks_inp = plain_toks.repeat((n_active_beams, 1))
                # [n_active_beams, n_plain_toks + beam_seq_len]
                # print(plain_toks_inp.shape, beam_seq_batch.shape)
                toks_inp = torch.concatenate([plain_toks_inp, beam_seq_batch], dim=-1)

            # [n_active_beams, n_chunks, d_model]
            chunks_emb_inp = chunks_emb.repeat((n_active_beams, 1, 1))
            # chunks_emb_inp = chunks_emb_inp[:, :1]

            # [n_active_beams, n_plain_toks + beam_seq_len]
            toks_inp_mask = toks_inp != self.tkz.pad_token_id
            # [n_active_beams, n_chunks]
            chunks_mask = torch.ones((n_active_beams, n_chunks), dtype=torch.bool, device=self.device)
            # [n_active_beams, n_chunks + n_plain_toks + beam_seq_len]
            inp_mask = torch.concatenate([chunks_mask, toks_inp_mask], dim=1)
            token_type_ids = None
            if self.cfg.token_types_for_embs:
                token_type_ids = torch.zeros(inp_mask.shape, dtype=torch.int64, device=self.device)
                token_type_ids[:, 1:n_chunks] = 1
            # print(f'chunks_emb_inp: {chunks_emb_inp.shape}. toks_inp: {toks_inp.shape}, inp_mask: {inp_mask.shape} all: {inp_mask.all()}. token_type_ids: {token_type_ids.shape}')
            mix_out: BaseModelOutputWithPoolingAndCrossAttentions = self.dec_model(
                inputs_starting_embeds=chunks_emb_inp, input_ids=toks_inp, attention_mask=inp_mask, token_type_ids=token_type_ids,
            )

            # [n_active_beams, n_chunks + n_plain_toks + beam_seq_len, d_model]
            lhs = mix_out.last_hidden_state
            # [n_active_beams, d_model]
            out_logits = lhs[:, -1]
            if self.cfg.out_embs_type == EncmixOutEmbsType.Non:
                raise Exception(f'Out embeddings type {self.cfg.out_embs_type} is not supported')
            if self.cfg.out_embs_type == EncmixOutEmbsType.Inp:
                # [d_model, vocab_size]
                wemb_weights = self.dec_model.embeddings.word_embeddings.weight
                # [n_active_beams, vocab_size]
                out_logits = out_logits @ wemb_weights.T
            else:
                # [n_active_beams, vocab_size]
                out_logits = self.out_word_embeddings(out_logits)

            return out_logits

        beams = beam_search.run(run_inference)
        for beam in beams:
            print(self.tkz.decode(beam.tokens_cur))

        return beams[0].tokens_cur



