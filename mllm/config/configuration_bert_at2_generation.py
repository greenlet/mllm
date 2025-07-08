# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BertGeneration model configuration"""
import dataclasses
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union, Optional

import yaml
from transformers import PretrainedConfig, AutoConfig, BertTokenizer

from mllm.config.model import GenmixTrainDsType
from mllm.utils.utils import coalesce, bool_to_str


class BertAt2GenerationConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BertGenerationPreTrainedModel`]. It is used to
    instantiate a BertGeneration model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the BertGeneration
    [google/bert_for_seq_generation_L-24_bbc_encoder](https://huggingface.co/google/bert_for_seq_generation_L-24_bbc_encoder)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50358):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertGeneration`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often called feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.

    Examples:

    ```python
    >>> from transformers import BertGenerationConfig, BertGenerationEncoder

    >>> # Initializing a BertGeneration config
    >>> configuration = BertGenerationConfig()

    >>> # Initializing a model (with random weights) from the config
    >>> model = BertGenerationEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bert-at2-generation"

    def __init__(
        self,
        vocab_size=50358,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        position_embedding_type="absolute",
        use_cache=True,
        last_dec_to_all_enc_at2_enabled: bool = False,
        enc_at2_enabled: bool = False,
        dec_at2_enabled: bool = False,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.last_dec_to_all_enc_at2_enabled = last_dec_to_all_enc_at2_enabled
        self.enc_at2_enabled = enc_at2_enabled
        self.dec_at2_enabled = dec_at2_enabled


class EncEmbAggType(str, Enum):
    Non = 'non'
    Emb = 'cls'
    Avg = 'avg'
    Mat = 'mat'


class EncEmbExpType(str, Enum):
    Non = 'non'
    Mat = 'mat'


class EncoderAt2DecoderConfig(PretrainedConfig):
    model_type = "encoder-decoder"
    is_composition = True
    enc_inp_len: int = 0
    enc_emb_agg_type: EncEmbAggType
    enc_emb_exp_type: EncEmbExpType


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            "encoder" in kwargs and "decoder" in kwargs
        ), "Config has to be initialized with encoder and decoder config"
        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")
        decoder_config = kwargs.pop("decoder")
        decoder_model_type = decoder_config.pop("model_type")

        if '-at2-' in encoder_model_type:
            self.encoder = BertAt2GenerationConfig(**encoder_config)
            self.decoder = BertAt2GenerationConfig(**decoder_config)
        else:
            self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
            self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.is_encoder_decoder = True
        self.enc_inp_len = kwargs.get('enc_inp_len', self.enc_inp_len)

    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`EncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model configuration and
        decoder model configuration.

        Returns:
            [`EncoderDecoderConfig`]: An instance of a configuration object
        """
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)


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
