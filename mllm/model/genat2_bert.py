import os.path
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

from mllm.config.configuration_bert_at2_generation import Genat2Cfg
from mllm.model.at2_decoder import BertGenerationAt2Decoder, BertGenerationEncoder
from mllm.model.encoder_at2_decoder_bert import EncoderAt2DecoderModel


class Genat2Model(nn.Module):
    cfg: Genat2Cfg
    device: torch.device
    tkz: BertTokenizer
    model: EncoderAt2DecoderModel

    def __init__(self, cfg: Genat2Cfg, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg
        self.device = device if device is not None else torch.device('cpu')
        self.tkz = BertTokenizer.from_pretrained(self.cfg.pretrained_model_name)
        enc_model: BertGenerationEncoder = BertGenerationEncoder.from_pretrained(
            self.cfg.pretrained_model_name, config=self.cfg.bert.encoder, device_map=self.device,
        )
        dec_model: BertGenerationAt2Decoder = BertGenerationAt2Decoder.from_pretrained(
            self.cfg.pretrained_model_name, config=self.cfg.bert.decoder, device_map=self.device,
        )
        self.model = EncoderAt2DecoderModel(
            encoder=enc_model, decoder=dec_model,
        )
        self.model.config.decoder_start_token_id = self.tkz.cls_token_id
        self.model.config.pad_token_id = self.tkz.pad_token_id

    def load_weights_from_pretrained_encoder(self, pretrained_model_path: Path):
        if pretrained_model_path.is_dir():
            pretrained_model_path /= 'best.pth'
        print(f'Loading checkpoint with pretrained model from {pretrained_model_path}')
        pretrained_checkpoint = torch.load(pretrained_model_path)
        # model.load_state_dict(pretrained_checkpoint['model'], strict=False)
        print(list(pretrained_checkpoint['model'].keys()))
        prefix = 'enc_bert.bert_model.'
        prefix_len = len(prefix)
        model_chkpt = {}
        for k, v in pretrained_checkpoint['model'].items():
            if k.startswith(prefix):
                k = k[prefix_len:]
            if k.startswith('dec_pyr.') or k in ('pooler.dense.weight', 'pooler.dense.bias', 'embeddings.token_type_embeddings.weight'):
                continue
            model_chkpt[k] = v
        self.model.encoder.load_state_dict(model_chkpt, strict=True)
        del pretrained_checkpoint
        del model_chkpt

    def _to_toks(self, s: str, inp_len: Optional[int] = None) -> torch.Tensor:
        if inp_len is None:
            t = self.tkz(s, return_tensors='pt').input_ids.to(self.device)
            assert t[0][0] == self.tkz.cls_token_id and t[0][-1] == self.tkz.sep_token_id
            return t
        input_ids = self.tkz(s).input_ids
        assert input_ids[0] == self.tkz.cls_token_id and input_ids[-1] == self.tkz.sep_token_id
        # Excluding cls and sep tokens
        input_ids = input_ids[1:-1]
        chunks = []
        while input_ids:
            n = min(len(input_ids), inp_len - 2)
            ids = input_ids[:n]
            chunks.append([self.tkz.cls_token_id, *ids, self.tkz.sep_token_id])
            input_ids = input_ids[n:]

        res = np.full((len(chunks), inp_len), self.tkz.pad_token_id)
        for i in range(len(chunks)):
            ch = chunks[i]
            res[i][:len(ch)] = ch
        res = torch.from_numpy(res).to(self.device)
        return res

    def run_on_qna_txt(self, context: str, question: str, answer: str) -> torch.Tensor:
        n_ctx_max = 15000
        if len(context) > n_ctx_max:
            context = context[:n_ctx_max]
        prompt = f'Given the context, answer the question. <context>{context}</context><question>{question}</question>'
        # [n_prompt, inp_len]
        prompt_toks = self._to_toks(prompt, inp_len=self.cfg.inp_len)
        # [n_prompt, inp_len]
        prompt_mask = prompt_toks != self.tkz.pad_token_id

        # [1, n_ans]
        ans_toks = self._to_toks(answer)
        # [n_ans]
        ans_toks = ans_toks[0]
        n_ans = ans_toks.shape[0]
        if n_ans > self.cfg.max_out_toks:
            diff = n_ans - self.cfg.max_out_toks
            ans_toks = torch.concat([ans_toks[:-diff - 1], ans_toks[-1:]])
        # tgt_len = n_ans - 1
        # [tgt_len]
        target_ids = ans_toks[:-1]
        if target_ids[0] != self.tkz.cls_token_id:
            target_ids = F.pad(target_ids, (1, 0), 'constant', self.tkz.cls_token_id)
        # [1, tgt_len]
        target_ids = target_ids.unsqueeze(0)

        gen_out: Seq2SeqLMOutput = self.model(
            input_ids=prompt_toks, attention_mask=prompt_mask, decoder_input_ids=target_ids, use_cache=False,
        )
        # [1, tgt_len, n_vocab]
        gen_logits = gen_out.logits

        # [tgt_len, n_vocab]
        logits = gen_logits.view(-1, self.cfg.bert.decoder.vocab_size)
        # [tgt_len]
        labels = ans_toks[1:]
        # [tgt_len]
        loss = F.cross_entropy(logits, labels, reduction='none')
        # The last one is sep_token_id
        assert loss.shape[0] > 1
        loss_1, loss_2 = loss[:-1].mean(), loss[-1]
        w1, w2 = 50, 1
        loss = (loss_1 * w1 + loss_2 * w2) / (w1 + w2)
        return loss

    def run_on_sum_txt(self, title: str, text: str, summary: str) -> torch.Tensor:
        prompt = f'Summarize following text. Title: {title}. Text: {text}'
        # [n_prompt, inp_len]
        prompt_toks = self._to_toks(prompt, inp_len=self.cfg.inp_len)
        if self.cfg.max_inp_chunks > 0:
            prompt_toks = prompt_toks[:self.cfg.max_inp_chunks]
        # [n_prompt, inp_len]
        prompt_mask = prompt_toks != self.tkz.pad_token_id

        # [1, n_sum]
        sum_toks = self._to_toks(summary)
        if 0 < self.cfg.max_out_toks < sum_toks.shape[1]:
            sum_toks = sum_toks[:, :self.cfg.max_out_toks]
        # [n_sum]
        sum_toks = sum_toks[0]
        # tgt_len = n_sum - 1
        # [tgt_len]
        target_ids = sum_toks[:-1]
        if target_ids[0] != self.tkz.cls_token_id:
            target_ids = F.pad(target_ids, (1, 0), 'constant', self.tkz.cls_token_id)
        # [1, tgt_len]
        target_ids = target_ids.unsqueeze(0)

        gen_out: Seq2SeqLMOutput = self.model(
            input_ids=prompt_toks, attention_mask=prompt_mask, decoder_input_ids=target_ids, use_cache=False,
        )
        # [1, tgt_len, n_vocab]
        gen_logits = gen_out.logits

        # [tgt_len, n_vocab]
        dec_cfg = self.cfg.bert.decoder
        logits = gen_logits.view(-1, dec_cfg.vocab_size)
        # [tgt_len]
        labels = sum_toks[1:]
        # [tgt_len]
        loss = F.cross_entropy(logits, labels, reduction='none')
        # The last one is sep_token_id
        assert loss.shape[0] > 1
        if labels[-1] == self.tkz.sep_token_id:
            loss_1, loss_2 = loss[:-1].mean(), loss[-1]
            w1, w2 = 50, 1
            loss = (loss_1 * w1 + loss_2 * w2) / (w1 + w2)
        else:
            loss = loss.mean()
        return loss


def run_create_config():
    cfg = Genat2Cfg.create()
    print(cfg)


def run_create_model():
    cfg_fpath = Path(os.path.abspath('.')).parent / 'config/cfg/genat2_cfg_01_base.yaml'
    print(cfg_fpath, cfg_fpath.exists())
    cfg = Genat2Cfg.copy_override(cfg_fpath)
    model = Genat2Model(cfg)
    print(model)


def run_load_checkpoint():
    cfg_fpath = Path(os.path.abspath('.')).parent / 'config/cfg/genat2_cfg_01_base.yaml'
    pretrained_model_path = Path(os.path.expandvars('$HOME/data')) / 'train_mllm_encdec_bert' / 'encdecbert-20250131_223521-bert-base-uncased-d768-emb_cls-inp128-lrs7x1-enh_mmbb-step2-h12-dp0-t0.0'
    if not pretrained_model_path.exists():
        print(f'Path {pretrained_model_path} does not exist. Stopping.')
        return
    print(cfg_fpath, cfg_fpath.exists())
    cfg = Genat2Cfg.copy_override(cfg_fpath)
    model = Genat2Model(cfg, device=torch.device('cuda'))
    model.load_weights_from_pretrained_encoder(pretrained_model_path)
    print(model)


if __name__ == '__main__':
    run_create_config()
    # run_create_model()
    # run_load_checkpoint()


