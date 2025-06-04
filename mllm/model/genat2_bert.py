import dataclasses
import os.path
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import yaml
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

from mllm.config.configuration_bert_at2_generation import BertAt2GenerationConfig
from mllm.config.model import GenmixTrainDsType
from mllm.model.at2_decoder import BertGenerationAt2Decoder, BertGenerationEncoder
from mllm.model.encoder_at2_decoder_bert import EncoderAt2DecoderModel, EncoderAt2DecoderConfig
from mllm.utils.utils import coalesce, bool_to_str


@dataclass
class Genat2Cfg:
    inp_len: int
    pretrained_model_name: str
    max_inp_chunks: int
    max_out_toks: int
    bert: EncoderAt2DecoderConfig

    file_name = 'genat2_model_cfg.yaml'

    @classmethod
    def create(
            cls, inp_len: int = 128, pretrained_model_name: str = 'bert-base-uncased', max_inp_chunks: int = 10, max_out_toks: int = 50,
            encoder_enc_at2_enabled: bool = True, decoder_enc_at2_enabled: bool = True, decoder_dec_at2_enabled: bool = True,
            decoder_last_dec_to_all_enc_at2_enabled: bool = True,
    ) -> 'Genat2Cfg':
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        print(tokenizer)
        enc_model: BertGenerationEncoder = BertGenerationEncoder.from_pretrained(
            pretrained_model_name, bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.sep_token_id,
            enc_at2_enabled=encoder_enc_at2_enabled,
        )
        dec_model: BertGenerationAt2Decoder = BertGenerationAt2Decoder.from_pretrained(
            pretrained_model_name, add_cross_attention=True, is_decoder=True, bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id, use_cache=False,
            enc_at2_enabled=decoder_enc_at2_enabled, dec_at2_enabled=decoder_dec_at2_enabled, last_dec_to_all_enc_at2_enabled=decoder_last_dec_to_all_enc_at2_enabled,
        )
        model = EncoderAt2DecoderModel(
            encoder=enc_model, decoder=dec_model,
        )
        model.train()
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        bert_cfg = model.config

        return Genat2Cfg(
            inp_len=inp_len, pretrained_model_name=pretrained_model_name, max_inp_chunks=max_inp_chunks,
            max_out_toks=max_out_toks, bert=bert_cfg,
        )

    @classmethod
    def copy_override(
            cls, cfg: Union['Genat2Cfg', Path], inp_len: int = 0, pretrained_model_name: str = '', max_inp_chunks: int = 0, max_out_toks: int = 0,
            encoder_enc_at2_enabled: Optional[bool] = None, decoder_enc_at2_enabled: Optional[bool] = None, decoder_dec_at2_enabled: Optional[bool] = None,
            decoder_last_dec_to_all_enc_at2_enabled: Optional[bool] = None,
    ) -> 'Genat2Cfg':
        if not isinstance(cfg, Genat2Cfg):
            cfg = cls.load_from_yaml(cfg)

        inp_len = inp_len or cfg.inp_len
        pretrained_model_name = pretrained_model_name or cfg.pretrained_model_name
        max_inp_chunks = max_inp_chunks or cfg.max_inp_chunks
        max_out_toks = max_out_toks or cfg.max_out_toks
        bert_cfg = EncoderAt2DecoderConfig.from_dict(deepcopy(cfg.bert.to_dict()))
        enc_cfg: BertAt2GenerationConfig = bert_cfg.encoder
        dec_cfg: BertAt2GenerationConfig = bert_cfg.decoder
        enc_cfg.enc_at2_enabled = coalesce(encoder_enc_at2_enabled, enc_cfg.enc_at2_enabled)
        dec_cfg.enc_at2_enabled = coalesce(decoder_enc_at2_enabled, dec_cfg.enc_at2_enabled)
        dec_cfg.dec_at2_enabled = coalesce(decoder_dec_at2_enabled, dec_cfg.dec_at2_enabled)
        dec_cfg.last_dec_to_all_enc_at2_enabled = coalesce(decoder_last_dec_to_all_enc_at2_enabled, dec_cfg.last_dec_to_all_enc_at2_enabled)

        return Genat2Cfg(
            inp_len=inp_len, pretrained_model_name=pretrained_model_name, max_inp_chunks=max_inp_chunks,
            max_out_toks=max_out_toks, bert=bert_cfg,
        )

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data['bert'] = data['bert'].to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Genat2Cfg':
        data = deepcopy(data)
        data['bert'] = EncoderAt2DecoderConfig.from_dict(data['bert'])
        return cls(**data)

    def save_to_yaml(self, fpath: Union[Path, str]):
        fpath = Path(fpath)
        if fpath.is_dir():
            fpath /= self.file_name
        data = self.to_dict()
        with open(fpath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load_from_yaml(cls, fpath: Union[Path, str]) -> 'Genat2Cfg':
        fpath = Path(fpath)
        if fpath.is_dir():
            fpath /= cls.file_name
        with open(fpath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.to_dict() == other.to_dict()


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
        t = self.tkz(s, return_tensors='pt').input_ids.to(self.device)
        assert t[0][0] == self.tkz.cls_token_id and t[0][-1] == self.tkz.sep_token_id
        if inp_len is not None:
            t = t.reshape(-1)
            mod = len(t) % inp_len
            if mod > 0:
                pad_size = inp_len - mod
                t = F.pad(t, (0, pad_size), 'constant', self.tkz.pad_token_id)
            t = t.reshape(-1, inp_len)
        return t

    def run_on_qna_txt(self, context: str, question: str, answer: str) -> torch.Tensor:
        context = f'<context>{context}</context>'
        question = f'<question>{question}</question>'
        # [n_ctx, inp_len]
        ctx_toks = self._to_toks(context, inp_len=self.cfg.inp_len)
        # [n_qst, inp_len]
        qst_toks = self._to_toks(question, inp_len=self.cfg.inp_len)
        n_cq = ctx_toks.shape[0] + qst_toks.shape[0]
        if n_cq > self.cfg.max_inp_chunks:
            diff = n_cq - self.cfg.max_inp_chunks
            ctx_toks = ctx_toks[:ctx_toks.shape[0] - diff]

        # n_cq = n_ctx + n_qst
        # [n_cq, inp_len]
        cq_inp = torch.concat([ctx_toks, qst_toks])

        # [n_cq, inp_len]
        inp_mask = cq_inp != self.tkz.pad_token_id

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
            input_ids=cq_inp, attention_mask=inp_mask, decoder_input_ids=target_ids, use_cache=False,
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


def gen_prefpostfix_genat2(model_cfg: Genat2Cfg, train_ds_type: Optional[GenmixTrainDsType] = None) -> tuple[str, str]:
    prefix, postfix_parts = f'genat2', []

    bert_str = model_cfg.pretrained_model_name.replace('_', '_')
    postfix_parts.append(bert_str)

    enc_cfg: BertAt2GenerationConfig = model_cfg.bert.encoder
    dec_cfg: BertAt2GenerationConfig = model_cfg.bert.decoder
    postfix_parts.append(f'd{dec_cfg.hidden_size}')

    postfix_parts.append(f'inp{model_cfg.inp_len}')

    postfix_parts.append(f'eeat2{bool_to_str(enc_cfg.enc_at2_enabled, cap=False)}')
    postfix_parts.append(f'deat2{bool_to_str(dec_cfg.enc_at2_enabled, cap=False)}')
    postfix_parts.append(f'ddat2{bool_to_str(dec_cfg.dec_at2_enabled, cap=False)}')
    postfix_parts.append(f'dldeat2{bool_to_str(dec_cfg.last_dec_to_all_enc_at2_enabled, cap=False)}')

    if train_ds_type is not None:
        postfix_parts.append(f'ds_{train_ds_type.value}')

    if model_cfg.max_inp_chunks > 0:
        postfix_parts.append(f'maxi{model_cfg.max_inp_chunks}')

    if model_cfg.max_out_toks > 0:
        postfix_parts.append(f'maxo{model_cfg.max_out_toks}')

    postfix = '-'.join(postfix_parts)
    return prefix, postfix


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


