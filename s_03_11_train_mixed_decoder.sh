
# code_path=$HOME/prog
# data_path=$HOME/data
# mllm_src_path=$code_path/mllm
code_path=$AZUREML_CR_EXECUTION_WORKING_DIR_PATH
data_path=$code_path/data
mllm_src_path=$code_path

#wiki_ds_name=20200501.en
wiki_ds_name=20220301.en

config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fname=mixed_decoder_cfg_01.yaml

model_cfg_fpath=$config_dir_path/$model_cfg_fname
train_root_path=$data_path/train_mllm_encdec_bert

bert_model_name=bert-base-uncased
bert_emb_type=cls
inp_len=128

# Decoder type: gpt2 or bertdec
decoder_type=gpt2
decoder_model_name=gpt2
# decoder_type=bertdec
# decoder_model_name=bert-base-uncased

# Compound decoder spec: <family>-<size>[-instruct]-<precision>
# Examples:
#   decoder_spec=qwen2.5-1.5B-fp32          # full fp32, single 1.5B Qwen2.5 base
#   decoder_spec=qwen2.5-1.5B-fp16          # AMP fp16 + GradScaler (V100-friendly, DDP only)
#   decoder_spec=qwen2.5-1.5B-bf16          # bf16 (preferred with FSDP)
#   decoder_spec=qwen2.5-1.5B-instruct-fp16 # instruct variant, AMP fp16
#   decoder_spec=qwen2.5-0.5B-fp32          # smoke-test size
#   decoder_spec=qwen3-0.6B-fp32
#   decoder_spec=gpt2-fp32                  # equivalent to the legacy gpt2 path
# When decoder_spec is non-empty it overrides decoder_type / decoder_model_name above.
decoder_spec=qwen2.5-1.5B-bf16

# Parallelism: 'ddp' (default, full replica per rank) or 'fsdp' (shards params/grads/
# optimizer state across ranks; required to fit Qwen2.5-1.5B+BERT on 32GB GPUs).
# fsdp_shard: 'full' (FULL_SHARD across all ranks, min memory) or 'hybrid' (HYBRID_SHARD,
# shard within node and replicate across; higher throughput, higher memory).
# FSDP path requires bf16 or fp32 (fp16 is unsupported here).
parallel=fsdp
fsdp_shard=full

# pip install datasets==3.6.0
train_ds_type=cite

# train_ds_type=qnasqv2
# train_ds_type=qnaall
train_ds_type=qnaans
# train_ds_type=next

min_next_toks=64

max_seq_len=400
freeze_encoder=false
# use_sep=true
use_sep=false
prompt_all=false
emb_exp_rate=4
emb_win_min_size=2
emb_win_max_size=6

decoder_only=false
# decoder_only=true
# inp_len * emb_win_max_size * emb_exp_rate
# max_seq_len=$((max_seq_len + inp_len * emb_win_max_size * emb_exp_rate))

mask_tokens=false
mask_sep_freq=0.5
mask_sep_frac=0.15
mask_seq_freq=0.5
mask_seq_max_frac=0.2
mask_seq_max_len=20
mask_n_last_toks=0

pretrained_encdec_model_path=$train_root_path/encdecbert-20260110_193915-bertbaseuncased-d768-embCls-inp128-lrs7x1-enhMmbb-step2-h12-dp0-t0.0
# pretrained_mixed_decoder_model_path=$train_root_path/mixeddecoder-20260304_105309-pre_encdecbert20260110193915-bertbaseuncased-d768-embEncCls-inp128-decGpt2-decmgpt2-msl384-sepT-pallF-eer4-ewn10x10-frzencF-trn_lr5e-05_bs30
# pretrained_mixed_decoder_model_path=$train_root_path/mixeddecoder-20260316_221645-pre_mixeddecoder20260304105309-bertbaseuncased-d768-embEncCls-inp128-decBertbaseuncased-msl384-sepT-pallF-eer4-ewn10x10-frzencF-dsCite-trn_lr5e-05_bs40
# pretrained_mixed_decoder_model_path=$train_root_path/mixeddecoder-20260319_130017-pre_mixeddecoder20260316221645-bertbaseuncased-d768-embEncCls-inp128-decBertbaseuncased-msl384-sepT-pallF-eer4-ewn10x10-frzencF-dsCite-msk_sep0.5x0.15_seq0.5x0.2x20_last0-trn_lr5e-05_bs40
# pretrained_mixed_decoder_model_path=$train_root_path/mixeddecoder-20260429_091845-pre_encdecbert20260110193915-bertbaseuncased-d768-embEncCls-inp128-decGpt2-msl384-sepF-pallF-eer4-ewn2x4-frzencF-dsCite-trn_lr5e-05_bs30
pretrained_mixed_decoder_model_path=$train_root_path/mixeddecoder-20260523_180218-pre_encdecbert20260110193915-bertbaseuncased-d768-embEncCls-inp128-decQwen2.51.5b-msl400-dtypeBf16-sepF-pallF-eer4-ewn2x6-frzencF-dsCite-msk_sep0.5x0.15_seq0.5x0.2x20_last0-trn_lr5e-05_bs20_attdp0.1
# train_subdir=last

# device=cpu
# epochs=5
# train_epoch_steps=20
# val_epoch_steps=20
# docs_batch_size=5
# world_size=1

device=cuda
epochs=700
train_epoch_steps=500
val_epoch_steps=50
# docs_batch_size=40
# docs_batch_size=30
docs_batch_size=20
# docs_batch_size=15
# docs_batch_size=5
world_size=4


learning_rate=0.00005
# If > 0, overrides the current learning rate by rebuilding optimizer and scheduler
# from scratch (any restored optimizer/scheduler state from checkpoint is discarded).
learning_rate_override=0
random_seed=200

optimizer_name='AdamW'
optimizer_params='{}'
# optimizer_name='Adam'
# optimizer_params='{}'
learning_rate_scheduler_name='ReduceLROnPlateau'
learning_rate_scheduler_params='{"mode": "min", "factor": 0.5, "patience": 5, "threshold": 1e-6, "min_lr": 1e-8}'

# optimizer_name='AdamW'
# optimizer_params='{"weight_decay": 0.01, "betas": [0.9, 0.98], "eps": 1e-8}'
# learning_rate_scheduler_name='CosineAnnealingWarmRestarts'
# learning_rate_scheduler_params='{"T_0": 30, "T_mult": 2, "eta_min": 1e-7}'

# Regularization knobs (all default to no-op).
# Recommended "Qwen2.5-1.5B regularized" preset:
#   weight_decay_decoder=0.1
#   weight_decay_other=0.01
#   llrd_decay=0.9
#   attention_dropout=0.1
#   label_smoothing=0.1
#   max_grad_norm=1.0
weight_decay_decoder=0.0
weight_decay_other=0.0
llrd_decay=1.0
attention_dropout=0.1
label_smoothing=0.0
max_grad_norm=0.0


export PYTHONPATH=$PYTHONPATH:$mllm_src_path

export NCCL_DEBUG=WARN          # downgrade INFO noise but keep warnings/errors
# DETAIL wraps every collective with ProcessGroupWrapper (gloo monitoredBarrier +
# per-call fingerprint allreduce) and has been the source of spurious
# "Connection closed by peer" failures on this cluster. Use INFO; switch back to
# DETAIL only when diagnosing DDP correctness issues (mark-ready-twice, missing
# gradients, etc.).
# export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=OFF
# CUDA_LAUNCH_BLOCKING=1 forces every CUDA op to be synchronous, which kills the
# comm/compute overlap that FSDP relies on (per-layer all-gather meant to run
# concurrently with the previous layer's matmul). It also slows DDP a bit, but
# the impact on FSDP is severe (~3-4x). Re-enable only for debugging CUDA errors.
# export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1


cd "$mllm_src_path" || exit 1
python s_03_11_train_mixed_decoder.py \
  --data-path $data_path \
  --wiki-ds-name $wiki_ds_name \
  --train-root-path $train_root_path \
  --train-subdir "$train_subdir" \
  --model-cfg-fpath $model_cfg_fpath \
  --bert-model-name $bert_model_name \
  --bert-emb-type $bert_emb_type \
  --inp-len $inp_len \
  --decoder-type $decoder_type \
  --decoder-model-name $decoder_model_name \
  --decoder-spec "$decoder_spec" \
  --max-seq-len $max_seq_len \
  --freeze-encoder $freeze_encoder \
  --use-sep $use_sep \
  --prompt-all $prompt_all \
  --decoder-only $decoder_only \
  --emb-exp-rate $emb_exp_rate \
  --emb-win-min-size $emb_win_min_size \
  --emb-win-max-size $emb_win_max_size \
  --train-ds-type $train_ds_type \
  --min-next-toks $min_next_toks \
  --mask-tokens $mask_tokens \
  --mask-sep-freq $mask_sep_freq \
  --mask-sep-frac $mask_sep_frac \
  --mask-seq-freq $mask_seq_freq \
  --mask-seq-max-frac $mask_seq_max_frac \
  --mask-seq-max-len $mask_seq_max_len \
  --mask-n-last-toks $mask_n_last_toks \
  --docs-batch-size $docs_batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --learning-rate-override $learning_rate_override \
  --optimizer-name $optimizer_name \
  --optimizer-params "$optimizer_params" \
  --learning-rate-scheduler-name $learning_rate_scheduler_name \
  --learning-rate-scheduler-params "$learning_rate_scheduler_params" \
  --weight-decay-decoder $weight_decay_decoder \
  --weight-decay-other $weight_decay_other \
  --llrd-decay $llrd_decay \
  --attention-dropout $attention_dropout \
  --label-smoothing $label_smoothing \
  --max-grad-norm $max_grad_norm \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --random-seed $random_seed \
  --pretrained-encdec-model-path "$pretrained_encdec_model_path" \
  --pretrained-mixed-decoder-model-path "$pretrained_mixed_decoder_model_path" \
  --world-size $world_size \
  --parallel $parallel \
  --fsdp-shard $fsdp_shard

