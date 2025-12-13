from pathlib import Path
from typing import Optional, Callable

from pydantic import BaseModel, Field

TOKENIZER_CFG_FNAME = 'tokenizer_cfg.yaml'
ENCDEC_MODEL_CFG_FNAME = 'encdec_model_cfg.yaml'
RANKER_MODEL_CFG_FNAME = 'ranker_model_cfg.yaml'
ENCDEC_HG_MODEL_CFG_FNAME = 'encdec_hg_model_cfg.yaml'
RANKER_HG_MODEL_CFG_FNAME = 'ranker_hg_model_cfg.yaml'
ENCDEC_BERT_MODEL_CFG_FNAME = 'encdec_bert_model_cfg.yaml'
ENCDEC_GRAPH_BERT_MODEL_CFG_FNAME = 'encdec_graph_bert_model_cfg.yaml'
RANKER_BERT_MODEL_CFG_FNAME = 'ranker_bert_model_cfg.yaml'
ENCMIX_BERT_MODEL_CFG_FNAME = 'encmix_bert_model_cfg.yaml'
GENMIX_BERT_MODEL_CFG_FNAME = 'genmix_bert_model_cfg.yaml'
GENMIXEMB_BERT_MODEL_CFG_FNAME = 'genmixemb_model_cfg.yaml'

mask_tokens_ARG = '--mask-tokens', 'Mask input tokens'
next_tok_pred_ARG = '--next-tok-pred', 'Predict next token'
masked_loss_for_encoder_ARG = '--masked-loss-for-encoder', 'Add masked loss for encoder'

ARG_TRUE_VALUES = ('true', 't', 'yes', 'y', '1')
ARG_FALSE_VALUES = ('false', 'f', 'no', 'n', '0')
ARG_TRUE_VALUES_STR = f'[{",".join(ARG_TRUE_VALUES)}]'
ARG_FALSE_VALUES_STR = f'[{",".join(ARG_FALSE_VALUES)}]'

def is_arg_true(name: str, val) -> bool:
    val = val.lower()
    if val in ARG_TRUE_VALUES:
        return True
    if val in ARG_FALSE_VALUES:
        return False
    raise Exception(f'{name} value can either have value from {ARG_TRUE_VALUES_STR} which means True '
                    f'or {ARG_FALSE_VALUES_STR} to be False (case insensitive). Value given: "{val}"')


def create_bool_str_field(arg_name: str, desc_part: str, default_value: str = 'true') -> Field:
    ref_name = arg_name.strip('-').replace('-', '_').upper()
    field = Field(
        default_value,
        required=False,
        description=f'Boolean flag. {desc_part}. ' \
            f'{ref_name} can take value from {ARG_TRUE_VALUES_STR} to be True or {ARG_FALSE_VALUES_STR} to be False. '
            f'Case insensitive.',
        cli=(arg_name,),
    )
    return field


class ArgsTokensChunksTrain(BaseModel):
    ds_dir_path: Path = Field(
        None,
        required=False,
        description='Dataset directory path. Must contain .csv and .np files with tokenized text.',
        cli=('--ds-dir-path',),
    )
    train_root_path: Path = Field(
        ...,
        required=True,
        description='Path to train root directory. New train subdirectory will be created within each new run.',
        cli=('--train-root-path',),
    )
    train_subdir: str = Field(
        '',
        required=False,
        description='Train subdirectory. Can have values: "last", "<subdirectory-name>". When set to "last", '
            'last subdirectory of TRAIN_ROOT_PATH containing training snapshot will be taken.',
        cli=('--train-subdir',)
    )
    tokenizer_cfg_fpath: Path = Field(
        ...,
        required=True,
        description='Path to tokenizer config Yaml file.',
        cli=('--tokenizer-cfg-fpath',),
    )
    model_cfg_fpath: Path = Field(
        ...,
        required=True,
        description='Path to ranker model config Yaml file.',
        cli=('--model-cfg-fpath',),
    )
    model_level: int = Field(
        ...,
        required=True,
        description='Model level. 0 - start from tokens and produce embeddins_0. k - start from embeddings from level k - 1 '
                    'and produce embeddings_k.',
        cli=('--model-level',),
    )
    n_enc_layers: int = Field(
        0,
        required=True,
        description='Number of Encoder transformer layers. When greater then 0 will override corresponding value in model\'s config.',
        cli=('--n-enc-layers',),
    )
    n_dec_layers: int = Field(
        0,
        required=True,
        description='Number of Decoder transformer layers. When greater then 0 will override corresponding value in model\'s config.',
        cli=('--n-dec-layers',),
    )
    dec_with_vocab_decoder: str = Field(
        'true',
        required=True,
        description='Boolean flag determining whether Encode-Decoder level 0 model last layer should be VocabDecoder. ' \
            f'Can have one value from {ARG_TRUE_VALUES_STR} to be True or {ARG_FALSE_VALUES_STR} to be False. (default: true)',
        cli=('--dec-with-vocab-decoder',),
    )
    @property
    def dec_with_vocab_decoder_bool(self) -> bool:
        return is_arg_true('--dec-with-vocab-decoder', self.dec_with_vocab_decoder)

    docs_batch_size: int = Field(
        3,
        required=False,
        description='Documents batch size. Must be greater or equal than 2.',
        cli=('--docs-batch-size',),
    )
    max_chunks_per_doc: int = Field(
        3,
        required=False,
        description='Maximum number of consecutive chunks per document taken in each butch. '
                    'Batch chunk max size will be DOCS_BATCH_SIZE * MAX_CHUNKS_PER_DOC.',
        cli=('--max-chunks-per-doc',),
    )
    device: str = Field(
        'cpu',
        required=False,
        description='Device to run training on. Can have values: "cpu", "cuda"',
        cli=('--device',)
    )
    epochs: int = Field(
        None,
        required=True,
        description='Number of training epochs.',
        cli=('--epochs',),
    )
    learning_rate: float = Field(
        0.001,
        required=False,
        description='Initial learning rate of the training process.',
        cli=('--learning-rate',)
    )
    train_epoch_steps: Optional[int] = Field(
        None,
        required=False,
        description='Number of training steps per epoch.',
        cli=('--train-epoch-steps',),
    )
    val_epoch_steps: Optional[int] = Field(
        None,
        required=False,
        description='Number of validation steps per epoch.',
        cli=('--val-epoch-steps',),
    )
    pretrained_model_path: Optional[Path] = Field(
        None,
        required=False,
        description='Path to pretrained model weights.',
        cli=('--pretrained-model-path',),
    )
    emb_chunk_size: Optional[int] = Field(
        100,
        required=False,
        description='Number of tokens in chunk converted to a single embedding vector.',
        cli=('--embs-chunk-size',),
    )


def get_pretrained_model_path(args_pretrained_model_path: Optional[Path], weights_fname: str = 'best.pth') -> Optional[Path]:
    if args_pretrained_model_path and args_pretrained_model_path.name:
        pretrained_model_path = args_pretrained_model_path
        if not pretrained_model_path.is_file():
            pretrained_model_path /= weights_fname
    else:
        pretrained_model_path = None
    return pretrained_model_path


