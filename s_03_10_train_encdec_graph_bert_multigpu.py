import json
import os
from pathlib import Path
from pprint import pprint
from typing import Any, Optional
import shutil

from datasets import Dataset
from pydantic import BaseModel, Field, validator
from pydantic_cli import run_and_exit
from pydantic_yaml import parse_yaml_file_as, to_yaml_file
import torch
import torch.utils.tensorboard as tb
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from transformers import AutoTokenizer

from mllm.config.model import EmbRnnInputOrder, EncdecCiteEmbsTargetType, EncdecCiteToksTargetType, EncdecGraphBertCfg, EncdecMiddleType, HgEnhanceType, EncdecBertCfg, copy_override_encdec_bert_cfg, BertEmbType, copy_override_encdec_graph_bert_cfg, \
    gen_prefpostfix_encdec_bert, gen_prefpostfix_encdec_graph_bert
from mllm.exp.args import ENCDEC_BERT_MODEL_CFG_FNAME, create_bool_str_field, get_pretrained_model_path, is_arg_true, mask_tokens_ARG, next_tok_pred_ARG, \
    share_enc_dec_proj_weights_ARG
from mllm.model.encdec_ranker_hg import EncdecBertAgg, EncdecGraphBert
from mllm.model.losses import LossesStats
from mllm.train.encdec_graph_bert import MaskedCiteDataset, create_masked_cite_dataloader, load_split_wiki_dataset
from mllm.train.mask_utils import MaskCfg
from mllm.train.utils import find_create_train_path, log_weights_grads_stats
from mllm.train.encdec_bert import create_dataloader_iter, load_masked_wiki_dataset
from mllm.utils.utils import instantiate_class, instantiate_torch_lr_scheduler, instantiate_torch_optimizer, parse_dict_str, rethrow


class ArgsEncdecGraphBertMultigpuTrain(BaseModel):
    data_path: Path = Field(
        ...,
        description='Root data path. Must contain subpath `wikipedia/WIKI_DS_NAME` with Wikipedia dataset.',
        cli=('--data-path',),
    )
    wiki_ds_name: str = Field(
        '20200501.en',
        description='Wikipedia dataset name of the format YYYYMMDD.LANG, for example: 20220301.en',
        cli=('--wiki-ds-name',),
    )
    train_root_path: Path = Field(
        ...,
        description='Path to train root directory. New train subdirectory will be created within each new run.',
        cli=('--train-root-path',),
    )
    train_subdir: str = Field(
        '',
        description='Train subdirectory. Can have values: "last", "<subdirectory-name>". When set to "last", '
            'last subdirectory of TRAIN_ROOT_PATH containing training snapshot will be taken.',
        cli=('--train-subdir',)
    )
    model_cfg_fpath: Path = Field(
        ...,
        description='Path to EncdecHg model config Yaml file.',
        cli=('--model-cfg-fpath',),
    )
    bert_model_name: str = Field(
        'bert-base-uncased',
        description='Pretrained BERT model name. Must be a model from Huggingface models hub (bert-base-*, bert-large-*).',
        cli=('--bert-model-name',),
    )
    bert_emb_type: BertEmbType = Field(
        BertEmbType.Cls,
        description=f'Bert embedding type. Can have values: {list(x.value for x in BertEmbType)}',
        cli=('--bert-emb-type',),
    )
    inp_len: int = Field(
        ...,
        description='Input tokens number. Must be a power of 2. INP_LEN = 2^k will produce model with k layers.',
        cli=('--inp-len',),
    )
    dec_enhance_type: HgEnhanceType = Field(
        HgEnhanceType.Matmul,
        description=f'Decoder layer enhance type. Can have values: {list(x.value for x in HgEnhanceType)}',
        cli=('--dec-enhance-type',),
    )
    dec_n_layers: int = Field(
        0,
        description='Decoder number of layers.',
        cli=('--dec-n-layers',),
    )
    dec_n_similar_layers: int = Field(
        ...,
        description='Number of consecutive similar attention layers for each decoder level.',
        cli=('--dec-n-similar-layers',),
    )
    dec_dropout_rate: float = Field(
        0.0,
        required=False,
        description='Decoder dropout rate.',
        cli=('--dec-dropout-rate',),
    )

    share_enc_dec_proj_weights_STR: str = create_bool_str_field(*share_enc_dec_proj_weights_ARG)
    @property
    def share_enc_dec_proj_weights(self) -> bool:
        return is_arg_true(share_enc_dec_proj_weights_ARG[0], self.share_enc_dec_proj_weights_STR)

    emb_middle_type: EncdecMiddleType = Field(
        EncdecMiddleType.Graph,
        description=f'Embedding processing model middle type. Can have values: {list(x.value for x in EncdecMiddleType)}',
        cli=('--emb-middle-type',),
    )
    n_graph_layers: int = Field(
        1,
        description='Number of graph layers.',
        cli=('--n-graph-layers',),
    )
    gnn_hidden_dim: int = Field(
        -1,
        description='Hidden dimension size for GNN layers. If set to -1, defaults to model dimension.',
        cli=('--gnn-hidden-dim',),
    )
    gnn_conv_name: str = Field(
        'GCNConv',
        description='GNN convolution layer class name. Must be one of the layers supported by PyTorch Geometric library ' \
            f'(https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers).',
        cli=('--gnn-conv-name',),
    )
    gnn_conv_params: dict = Field(
        {},
        description='GNN convolution layer parameters as a dictionary.',
        cli=('--gnn-conv-params',),
    )
    @validator('gnn_conv_params', pre=True)
    def parse_gnn_conv_params(cls, v):
        return parse_dict_str(v, 'gnn_conv_params')
    n_emb_attn_layers: int = Field(
        1,
        description='Number of embedding attention layers in the middle model.',
        cli=('--n-emb-attn-layers',),
    )

    emb_mlp_window_size: int = Field(
        3,
        description='Window size for MLP as the middle embedding model.',
        cli=('--emb-mlp-window-size',),
    )
    emb_mlp_n_window_layers: int = Field(
        1,
        description='Number of window layers for MLP as the middle embedding model.',
        cli=('--emb-mlp-n-window-layers',),
    )
    emb_mlp_n_out_layers: int = Field(
        1,
        description='Number of output layers for MLP as the middle embedding model.',
        cli=('--emb-mlp-n-out-layers',),
    )
    emb_mlp_act_fn: str = Field(
        'gelu',
        description='Activation function for MLP as the middle embedding model.',
        cli=('--emb-mlp-act-fn',),
    )

    emb_rnn_n_layers: int = Field(
        1,
        description='Number of stacked recurrent layers for RNN as the middle embedding model.',
        cli=('--emb-rnn-n-layers',),
    )
    emb_rnn_hidden_dim: int = Field(
        -1,
        description='Hidden dimension size for RNN layers. If non positive, defaults to model dimension.',
        cli=('--emb-rnn-hidden-dim',),
    )
    emb_rnn_input_order: EmbRnnInputOrder = Field(
        EmbRnnInputOrder.ContextPrompts,
        description=f'Order of input sequences for the RNN: "{EmbRnnInputOrder.PromptsContext.value}" for prompts-context or "{EmbRnnInputOrder.ContextPrompts.value}" for context-prompts.',
        cli=('--emb-rnn-input-order',),
    )
    emb_rnn_cell_name: str = Field(
        'LSTM',
        description='RNN cell class name. Must be one of: RNN, LSTM, GRU.',
        cli=('--emb-rnn-cell-name',),
    )
    emb_rnn_cell_params: dict = Field(
        {},
        description='RNN cell parameters as a dictionary.',
        cli=('--emb-rnn-cell-params',),
    )
    @validator('emb_rnn_cell_params', pre=True)
    def parse_emb_rnn_cell_params(cls, v):
        return parse_dict_str(v, 'emb_rnn_cell_params')

    mask_tokens_STR: str = create_bool_str_field(*mask_tokens_ARG)
    @property
    def mask_tokens(self) -> bool:
        return is_arg_true(mask_tokens_ARG[0], self.mask_tokens_STR)

    mask_sep_freq: float = Field(
        ...,
        description='Sparse mask frequency from 0 to 1. When MASK_SEP_FREQ=0.2 this type of mask will be applied in 20% of cases randomly. '
                    'Must hold: 0 <= MASK_SEP_FREQ and MASK_SEP_FREQ + MASK_SEQ_FREQ <= 1',
        cli=('--mask-sep-freq',),
    )
    mask_sep_frac: float = Field(
        ...,
        description='Fraction of the input to mask using sparse masking.',
        cli=('--mask-sep-frac',),
    )
    mask_seq_freq: float = Field(
        ...,
        description='Sequential mask frequency from 0 to 1. When MASK_SEQ_FREQ=0.2 this type of mask will be applied in 20% of cases randomly. '
                    'Must hold: 0 <= MASK_SEQ_FREQ and MASK_SEP_FREQ + MASK_SEQ_FREQ <= 1',
        cli=('--mask-seq-freq',),
    )
    mask_seq_max_frac: float = Field(
        ...,
        description='Fraction of the input to calculate maximum length of tokens sequence to mask. Resulting value is combined '
                    'with MASK_SEQ_MAX_LEN using min() function.',
        cli=('--mask-seq-max-frac',),
    )
    mask_seq_max_len: int = Field(
        ...,
        description='Maximum length of tokens sequence to mask. Combined with value derived from MASK_SEQ_MAX_FRAC using min() function.',
        cli=('--mask-seq-max-len',),
    )
    mask_n_last_toks: int = Field(
        0,
        description='Number of last tokens to always mask. When 0, no tokens are masked.',
        cli=('--mask-n-last-toks',),
    )
    cite_toks_target_weight: float = Field(
        1.0,
        description='Citation tokens target weight in the loss function.',
        cli=('--cite-toks-target-weight',),
    )
    cite_toks_target_type: EncdecCiteToksTargetType = Field(
        EncdecCiteToksTargetType.All,
        description=f'Citation tokens target type in the loss function. Can have values: {list(x.value for x in EncdecCiteToksTargetType)}',
        cli=('--cite-toks-target-type',),
    )
    cite_toks_target_scale: float = Field(
        1.0,
        description='Citation tokens target scale/multiplier in the loss function.',
        cli=('--cite-toks-target-scale',),
    )
    cite_embs_target_weight: float = Field(
        1.0,
        description='Citation embeddings target weight in the loss function.',
        cli=('--cite-embs-target-weight',),
    )
    cite_embs_target_type: EncdecCiteEmbsTargetType = Field(
        EncdecCiteEmbsTargetType.Cos,
        description=f'Citation embeddings target type in the loss function. Can have values: {list(x.value for x in EncdecCiteEmbsTargetType)}',
        cli=('--cite-embs-target-type',),
    )
    cite_embs_target_scale: float = Field(
        1.0,
        description='Citation embeddings target multiplier in the loss function.',
        cli=('--cite-embs-target-multiplier',),
    )
    input_toks_target_weight: float = Field(
        1.0,
        description='Input tokens target weight in the loss function.',
        cli=('--input-toks-target-weight',),
    )
    input_toks_target_scale: float = Field(
        1.0,
        description='Input tokens target scale/multiplier in the loss function.',
        cli=('--input-toks-target-scale',),
    )
    docs_batch_size: int = Field(
        3,
        description='Documents batch size. Must be greater or equal than 2.',
        cli=('--docs-batch-size',),
    )
    encdec_freeze_epochs: int = Field(
        0,
        description='Number of epochs to freeze encoder and decoder (train only middle embedding model).',
        cli=('--encdec-freeze-epochs',),
    )
    device: str = Field(
        'cpu',
        description='Device to run training on. Can have values: "cpu", "cuda"',
        cli=('--device',)
    )
    epochs: int = Field(
        None,
        description='Number of training epochs.',
        cli=('--epochs',),
    )

    learning_rate: float = Field(
        0.001,
        description='Initial learning rate of the training process.',
        cli=('--learning-rate',)
    )
    optimizer_name: str = Field(
        'AdamW',
        description='Optimizer class name. Must be a valid PyTorch optimizer.',
        cli=('--optimizer-name',),
    )
    optimizer_params: dict = Field(
        {},
        description='Optimizer class parameters as a dictionary.',
        cli=('--optimizer-params',),
    )
    @validator('optimizer_params', pre=True)
    def parse_optimizer_params(cls, v):
        return parse_dict_str(v, 'optimizer_params')
    learning_rate_scheduler_name: str = Field(
        'ReduceLROnPlateau',
        description='Learning rate scheduler class name. Must be a valid PyTorch learning rate scheduler.',
        cli=('--learning-rate-scheduler-name',),
    )
    learning_rate_scheduler_params: dict = Field(
        {},
        description='Learning rate scheduler class parameters as a dictionary.',
        cli=('--learning-rate-scheduler-params',),
    )
    @validator('learning_rate_scheduler_params', pre=True)
    def parse_learning_rate_scheduler_params(cls, v):
        return parse_dict_str(v, 'learning_rate_scheduler_params')

    train_epoch_steps: Optional[int] = Field(
        None,
        description='Number of training steps per epoch.',
        cli=('--train-epoch-steps',),
    )
    val_epoch_steps: Optional[int] = Field(
        None,
        description='Number of validation steps per epoch.',
        cli=('--val-epoch-steps',),
    )
    random_seed: Optional[int] = Field(
        None,
        description='Random seed.',
        cli=('--random-seed',),
    )
    pretrained_encdec_model_path: Optional[Path] = Field(
        None,
        description='Path to EncdecBert model train directory (loads encoder/decoder only).',
        cli=('--pretrained-encdec-model-path',),
    )
    pretrained_encdecgraph_model_path: Optional[Path] = Field(
        None,
        description='Path to EncdecGraphBert model train directory (loads full model with strict mode).',
        cli=('--pretrained-encdecgraph-model-path',),
    )
    world_size: int = Field(
        1,
        description='Number of GPU instances to use for distributed training.',
        cli=('--world-size',),
    )



def setup(rank, world_size):
    if world_size <= 1:
        return
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # # We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
    # # such as CUDA, MPS, MTIA, or XPU.
    # acc = torch.accelerator.current_accelerator()
    # backend = torch.distributed.get_default_backend_for_device(acc)
    backend = 'nccl'
    # backend = 'c10d'
    # initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup():
    if not dist.is_initialized():
        return
    dist.destroy_process_group()


def train(rank: int, ds_train: Dataset, ds_val: Dataset, args: ArgsEncdecGraphBertMultigpuTrain):
    print(f'Running DDP training on rank {rank}.')
    def log(*msgs: Any, forall: bool = False):
        if rank == 0 or forall:
            print(*msgs)

    setup(rank, args.world_size)

    pretrained_encdec_model_path = get_pretrained_model_path(args.pretrained_encdec_model_path)
    pretrained_encdecgraph_model_path = get_pretrained_model_path(args.pretrained_encdecgraph_model_path)

    device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
    log(f'Using device {device}.')

    mask_cfg = None
    if args.mask_tokens:
        mask_cfg = MaskCfg(
            sep_freq=args.mask_sep_freq, sep_frac=args.mask_sep_frac, seq_freq=args.mask_seq_freq, seq_max_frac=args.mask_seq_max_frac,
            seq_max_len=args.mask_seq_max_len, n_last_toks=args.mask_n_last_toks,
        )

    model_cfg = parse_yaml_file_as(EncdecGraphBertCfg, args.model_cfg_fpath)
    model_cfg = copy_override_encdec_graph_bert_cfg(
        model_cfg, pretrained_model_name=args.bert_model_name, emb_type=args.bert_emb_type, inp_len=args.inp_len, dec_enhance_type=args.dec_enhance_type,
        dec_n_layers=args.dec_n_layers, dec_n_similar_layers=args.dec_n_similar_layers, dec_dropout_rate=args.dec_dropout_rate,
        share_enc_dec_proj_weights=args.share_enc_dec_proj_weights, middle_type=args.emb_middle_type,
        n_graph_layers=args.n_graph_layers, gnn_hidden_dim=args.gnn_hidden_dim, gnn_conv_name=args.gnn_conv_name, gnn_conv_params=args.gnn_conv_params,
        n_emb_attn_layers=args.n_emb_attn_layers, emb_mlp_window_size=args.emb_mlp_window_size,
        emb_mlp_n_window_layers=args.emb_mlp_n_window_layers, emb_mlp_n_out_layers=args.emb_mlp_n_out_layers,
        emb_mlp_act_fn=args.emb_mlp_act_fn,
        emb_rnn_n_layers=args.emb_rnn_n_layers, emb_rnn_hidden_dim=args.emb_rnn_hidden_dim,
        emb_rnn_input_order=args.emb_rnn_input_order, emb_rnn_cell_name=args.emb_rnn_cell_name, emb_rnn_cell_params=args.emb_rnn_cell_params,
        pretrained_encdec_model_path=pretrained_encdec_model_path, pretrained_encdecgraph_model_path=pretrained_encdecgraph_model_path,
        mask_cfg=mask_cfg,
        cite_toks_target_weight=args.cite_toks_target_weight, cite_toks_target_type=args.cite_toks_target_type, cite_toks_target_scale=args.cite_toks_target_scale,
        cite_embs_target_weight=args.cite_embs_target_weight, cite_embs_target_type=args.cite_embs_target_type, cite_embs_target_scale=args.cite_embs_target_scale,
        input_toks_target_weight=args.input_toks_target_weight, input_toks_target_scale=args.input_toks_target_scale, learning_rate=args.learning_rate,
        optimizer_name=args.optimizer_name, optimizer_params=args.optimizer_params,
        lrs_name=args.learning_rate_scheduler_name, lrs_params=args.learning_rate_scheduler_params,
        batch_size=args.docs_batch_size, encdec_freeze_epochs=args.encdec_freeze_epochs,
    )
    if rank == 0:
        pprint(model_cfg.dict())

    prefix, suffix = gen_prefpostfix_encdec_graph_bert(model_cfg)
    train_path = find_create_train_path(args.train_root_path, prefix, suffix, args.train_subdir, create=(rank == 0))
    log(f'train_path: {train_path}')

    last_checkpoint_path, best_checkpoint_path = train_path / 'last.pth', train_path / 'best.pth'
    checkpoint = None
    if args.train_subdir == 'last':
        assert last_checkpoint_path.exists(),\
            (f'train_subdir = `last`, train subdirectory found ({train_path.name}), '
             f'but file {last_checkpoint_path} does not exits.')

    if last_checkpoint_path.exists():
        log(f'Loading checkpoint from {last_checkpoint_path}')
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        log(f'Checkpoint with keys {list(checkpoint.keys())} loaded')
        chkpt_model_cfg = parse_yaml_file_as(EncdecBertCfg, train_path / ENCDEC_BERT_MODEL_CFG_FNAME)
        assert model_cfg == chkpt_model_cfg, f'{args.model_cfg_fpath} != {chkpt_model_cfg}'
    else:
        if rank == 0:
            to_yaml_file(train_path / ENCDEC_BERT_MODEL_CFG_FNAME, model_cfg)

    tkz = AutoTokenizer.from_pretrained(args.bert_model_name)

    log(model_cfg)
    model = EncdecGraphBert(model_cfg, tkz)

    model.load_pretrained(checkpoint)

    model.to(device)
    if args.world_size > 1:
        # find_unused_parameters = False
        find_unused_parameters = True
        ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=find_unused_parameters)
    else:
        ddp_model = model

    params = ddp_model.parameters()
    optimizer = instantiate_torch_optimizer(args.optimizer_name, params, lr=args.learning_rate, **args.optimizer_params)
    scheduler = instantiate_torch_lr_scheduler(args.learning_rate_scheduler_name, optimizer, **args.learning_rate_scheduler_params)

    last_epoch, val_loss_min, shuffle = -1, None, False
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        last_epoch = checkpoint['last_epoch']
        val_loss_min = checkpoint['val_loss_min']
        del checkpoint

    n_special_toks = 1000
    ds_train = MaskedCiteDataset(ds_train, tkz, max_seq_len=args.inp_len, n_special_toks=n_special_toks, mask_cfg=mask_cfg, device=device)
    ds_val = MaskedCiteDataset(ds_val, tkz, max_seq_len=args.inp_len, n_special_toks=n_special_toks, mask_cfg=mask_cfg, device=device)
    train_batch_it = create_masked_cite_dataloader(ds_train, batch_size=args.docs_batch_size)
    val_batch_it = create_masked_cite_dataloader(ds_val, batch_size=args.docs_batch_size)

    lr = optimizer.param_groups[0]['lr']
    log(f'Scheduler {scheduler.__class__.__name__} lr: {lr:0.10f}.')
    if rank == 0:
        tbsw = tb.SummaryWriter(log_dir=str(train_path))
        log(ddp_model)

        grad_log_interval, grad_log_step, grad_log_ind = args.train_epoch_steps // 10, 0, 0
        prev_train_steps = args.train_epoch_steps * (last_epoch + 1)
        if prev_train_steps > 0:
            grad_log_ind = (prev_train_steps - 1) // grad_log_interval + 1
    
    if args.world_size > 1:
        dist.barrier()
    for epoch in range(last_epoch + 1, args.epochs):
        ddp_model.train()
        train_losses = LossesStats()
        train_loss = 0.0
        if rank == 0:
            pbar = trange(args.train_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        else:
            pbar = range(args.train_epoch_steps)
        for _ in pbar:
            batch = next(train_batch_it)

            optimizer.zero_grad()
            if args.world_size > 1:
                loss_dict, _ = ddp_model.module.run_on_text_citation(batch, epoch=epoch)
            else:
                loss_dict, _ = ddp_model.run_on_text_citation(batch, epoch=epoch)
            loss = loss_dict['loss']
            loss.backward()

            if rank == 0:
                # Gradients must be available after loss.backward()
                if grad_log_ind % grad_log_interval == 0:
                    log_weights_grads_stats(grad_log_step, ddp_model, tbsw)
                    grad_log_step += 1
                grad_log_ind += 1

            optimizer.step()
            train_loss += loss.item()
            train_losses.update_dict(loss_dict)

            if rank == 0:
                losses_str = train_losses.to_cli_str(aggregate=False)
                pbar.set_postfix_str(f'Train. {losses_str}')

        train_loss /= args.train_epoch_steps
        if rank == 0:
            pbar.close()
            train_losses.log_to_tb('Train', epoch, tbsw)

        ddp_model.eval()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        val_losses = LossesStats()
        val_loss = 0.0
        if rank == 0:
            pbar = trange(args.val_epoch_steps, desc=f'Epoch {epoch}', unit='batch')
        else:
            pbar = range(args.val_epoch_steps)
        for _ in pbar:
            batch = next(val_batch_it)

            with torch.no_grad():
                if args.world_size > 1:
                    loss_dict, _ = ddp_model.module.run_on_text_citation(batch)
                else:
                    loss_dict, _ = ddp_model.run_on_text_citation(batch)
            loss = loss_dict['loss']

            val_loss += loss.item()
            val_losses.update_dict(loss_dict)

            if rank == 0:
                losses_str = val_losses.to_cli_str(aggregate=False)
                pbar.set_postfix_str(f'Val. {losses_str}')
        
        val_loss /= args.val_epoch_steps
        if rank == 0:
            pbar.close()
            val_losses.log_to_tb('Val', epoch, tbsw)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        if rank == 0:
            last_lr = scheduler.get_last_lr()[0]
            tbsw.add_scalar(f'{scheduler.__class__.__name__} lr', last_lr, epoch)

            print(f'Train mean loss: {train_loss:.6f}. Val mean loss: {val_loss:.6f}')
            train_losses_str = train_losses.to_cli_str(aggregate=True)
            val_losses_str = val_losses.to_cli_str(aggregate=True)
            print(f'Train mean losses: {train_losses_str}')
            print(f'Val mean losses: {val_losses_str}')
            print(f'Current lr: {last_lr:.10f}.')
            
            best = False
            if val_loss_min is None or val_loss < val_loss_min:
                val_loss_str = f'{val_loss_min}' if val_loss_min is None else f'{val_loss_min:.6f}'
                print(f'Val min loss change: {val_loss_str} --> {val_loss:.6f}')
                val_loss_min = val_loss
                best = True

            checkpoint = {
                'model': ddp_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'last_epoch': epoch,
                'val_loss_min': val_loss_min,
            }
            print(f'Saving checkpoint to {last_checkpoint_path}')
            torch.save(checkpoint, last_checkpoint_path)

            if best:
                print(f'New val loss minimum: {val_loss_min:.6f}. Saving checkpoint to {best_checkpoint_path}')
                shutil.copyfile(last_checkpoint_path, best_checkpoint_path)

    cleanup()


def main(args: ArgsEncdecGraphBertMultigpuTrain) -> int:
    print(args)
   
    mask_cfg = None
    if args.mask_tokens:
        mask_cfg = MaskCfg(
            sep_freq=args.mask_sep_freq, sep_frac=args.mask_sep_frac, seq_freq=args.mask_seq_freq, seq_max_frac=args.mask_seq_max_frac,
            seq_max_len=args.mask_seq_max_len, n_last_toks=args.mask_n_last_toks,
        )

    tkz = AutoTokenizer.from_pretrained(args.bert_model_name)
    ds_train, ds_val = load_split_wiki_dataset(
        data_path=args.data_path, tkz=tkz, max_seq_len=args.inp_len, val_split_ratio=0.05,
        mask_cfg=mask_cfg, random_seed=args.random_seed,
    )

    mp.spawn(train, args=(
        ds_train, ds_val, args,
    ), nprocs=args.world_size, join=True)


    return 0


if __name__ == '__main__':
    run_and_exit(
        ArgsEncdecGraphBertMultigpuTrain, main, 'Train Encoder-Graph-Decoder BERT model multi GPU training.', exception_handler=rethrow,
    )

