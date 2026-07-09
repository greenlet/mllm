
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
#   decoder_spec=qwen3-0.6B-fp16
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
# train_ds_types is a space-separated list. A single type trains on that dataset;
# multiple types are mixed in a round-robin cycle (see train_ds_batches_per_cycle)
# with per-type loss scaling (see train_ds_loss_weights).
train_ds_types="cite"

# train_ds_types="qnasqv2"
# train_ds_types="qnaall"
# train_ds_types="qnaans"
# train_ds_types="cite qnaans"   # compound: mix wiki citation + qna-with-answers
# train_ds_types="next"
# train_ds_types="keyval"   # key-value recall (requires prompt_all=false)
# train_ds_types="jsonfield"   # JSON field extraction (requires prompt_all=false)
# train_ds_types="jsonata"   # JSONata/jq-like selection+transform (requires prompt_all=false)
# train_ds_types="xmlxpath"   # XML/XPath extraction (requires prompt_all=false)
# train_ds_types="sql"   # SQL selection/aggregate extraction (requires prompt_all=false)
train_ds_types="cite keyval jsonfield jsonata xmlxpath sql"
train_ds_types="next"

# Per-type batches emitted per round-robin cycle and per-type loss weights.
# Each is a space-separated list of length 1 (broadcast to all types) or
# exactly len(train_ds_types). When normalize_train_ds_loss_weights=true the
# weights are rescaled to sum to 1.
train_ds_batches_per_cycle="1"
train_ds_loss_weights="1"
normalize_train_ds_loss_weights=false

min_next_toks=64

# --- Controlled next-token comparison (soft-context vs raw-context perplexity) ---
# When training with train_ds_types="next", set these > 0 to pin a fixed context
# window (N = next_fixed_win_size * (inp_len - 2) tokens) and a fixed target length
# K = next_fixed_target_toks. Train one run with decoder_only=false (soft context)
# and one with decoder_only=true (raw context) using the SAME values, then compare
# perplexity with s_03_13_eval_next_tok_ppl.py. 0 = off (legacy random windowing).
next_fixed_win_size=16
next_fixed_target_toks=512

# --- Multi-source next-token corpora (train_ds_types="next") ---
# Space-separated list of long-document corpora the next-token loader draws from.
# Known: wiki pg19 bookcorpusopen arxiv govreport gutenberg. Batches are drawn one
# source at a time with frequency proportional to each source's split size.
next_sources="wiki"
next_sources="pg19 bookcorpusopen arxiv govreport gutenberg"
# next_sources="wiki pg19 arxiv govreport gutenberg"
# Bridge prompt inserted between context and target. Set to "" to drop the prompt
# entirely (pure context+target).
next_prompt="Continue:"
next_prompt=""

# --- key-value recall (train_ds_type=keyval) difficulty knobs ---
keyval_min_pairs=4
keyval_max_pairs=32
keyval_value_max_words=3

# --- JSON field recall (train_ds_type=jsonfield) difficulty knobs ---
jsonfield_min_fields=4
jsonfield_max_fields=30
jsonfield_max_depth=3
jsonfield_max_array_len=4
jsonfield_value_max_words=3

# --- JSONata/jq selection+transform (train_ds_type=jsonata) knobs ---
jsonata_min_fields=4
jsonata_max_fields=28
jsonata_max_depth=3
jsonata_max_array_len=5
jsonata_value_max_words=3
jsonata_transform_prob=0.35

# --- XML/XPath extraction (train_ds_type=xmlxpath) knobs ---
xmlxpath_min_nodes=4
xmlxpath_max_nodes=52
xmlxpath_max_depth=4
xmlxpath_max_children=4
xmlxpath_value_max_words=3

# --- SQL selection/aggregate (train_ds_type=sql) knobs ---
sql_min_rows=4
sql_max_rows=16
sql_min_cols=3
sql_max_cols=5
sql_value_max_words=3
sql_transform_prob=0.30

# Structured datasets: pack each record to ~fill the inp_len token budget
# (cite-style dense chunks). fill_frac is the early-accept fraction of the budget.
structured_fill_to_budget=true
structured_fill_frac=0.90

max_seq_len=400
# For train_ds_types="next" the decoder sequence is
#   n_ctx + prompt + target = (next_fixed_win_size * emb_exp_rate) + prompt_len + next_fixed_target_toks
# e.g. 16*4 + 0 + 512 = 576, so max_seq_len must exceed that. 640 leaves headroom.
max_seq_len=640
freeze_encoder=false
# use_sep=true
use_sep=false
prompt_all=false
# prompt_first: when true, prompt goes BEFORE context embeddings: [prompt, (SEP), ctx_embs, target]
prompt_first=true
emb_exp_rate=4
emb_win_min_size=10
emb_win_max_size=10

# --- InteractiveExtractor (query-conditioned soft-token bridge) ---
# When use_interactive_extractor=true it replaces the plain emb_exp expansion:
# each context embedding -> ie_exp_rate decoder-space slots that VISIT the prompt
# through ie_num_layers attention blocks (ie_attn_type: cross|self).
use_interactive_extractor=false
ie_exp_rate=4
ie_num_layers=6
ie_attn_type=self
ie_n_heads=8
ie_mlp_ratio=4.0
ie_dropout=0.1
ie_norm_first=true
ie_max_ctx=64
ie_max_prompt_len=128
# false = prompt seen ONLY by the extractor VISIT step (not in the causal stream).
# Set false to break the in-stream shortcut: the citation must flow through the
# soft tokens, which keeps encoder/extractor gradients alive (anti vanishing).
ie_prompt_in_stream=false

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
# pretrained_mixed_decoder_model_path=$train_root_path/mixeddecoder-20260523_180218-pre_encdecbert20260110193915-bertbaseuncased-d768-embEncCls-inp128-decQwen2.51.5b-msl400-dtypeBf16-sepF-pallF-eer4-ewn2x6-frzencF-dsCite-msk_sep0.5x0.15_seq0.5x0.2x20_last0-trn_lr5e-05_bs20_attdp0.1
# pretrained_mixed_decoder_model_path=$train_root_path/mixeddecoder-20260615_083942-bertbaseuncased-d768-embEncCls-inp128-decQwen2.51.5b-msl400-dtypeBf16-sepF-pallF-ewn10x10-ieSelf_eer4_nl6_nh8_mlp4.0-ieStrmF-frzencF-dsCite-msk_sep0.5x0.15_seq0.5x0.2x20_last0-trn_lr5e-05_bs20_attdp0.1/
# train_subdir=last

# device=cpu
# epochs=5
# train_epoch_steps=20
# val_epoch_steps=20
# docs_batch_size=5
# world_size=1

device=cuda
epochs=700
# Number of initial epochs to keep the decoder weights frozen (encoder/extractor
# bridge still train; gradients flow through the decoder). 0 disables.
freeze_decoder_epochs=10
freeze_decoder_epochs=0
train_epoch_steps=500
val_epoch_steps=50
# docs_batch_size=40
# docs_batch_size=30
docs_batch_size=20
# docs_batch_size=15
docs_batch_size=8
# docs_batch_size is PER-GPU. Qwen2.5-1.5B + seq~576 + 152k-vocab cross-entropy
# OOMs on a 32GB V100 at 10; 4 leaves headroom while training still runs.
# docs_batch_size=4
world_size=4


learning_rate=0.00005
# learning_rate=0.00001
# If > 0, overrides the current learning rate by rebuilding optimizer and scheduler
# from scratch (any restored optimizer/scheduler state from checkpoint is discarded).
learning_rate_override=0
random_seed=200

optimizer_name='AdamW'
optimizer_params='{}'
# optimizer_name='Adam'
# optimizer_params='{}'
learning_rate_scheduler_name='ReduceLROnPlateau'
learning_rate_scheduler_params='{"mode": "min", "factor": 0.5, "patience": 8, "threshold": 1e-6, "min_lr": 1e-8}'

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
# label_smoothing=0.1
# max_grad_norm=1.0


# =============================================================================
# STAGED NEXT-TOKEN CURRICULUM (soft-token context)
# -----------------------------------------------------------------------------
# The blocks below OVERRIDE the scattered defaults above (bash: last assignment
# wins). Train Stage 1, then comment it out and uncomment Stage 2, train again,
# then Stage 3. Each stage keeps the soft-token count = win_size * emb_exp_rate
# roughly constant so max_seq_len stays valid, while progressively raising the
# compression ratio (N_ctx / N_soft) and the target length K.
#
# Common to all stages (task = next-token, soft context ON):
train_ds_types="next"
decoder_only=false
min_next_toks=64
# Regularization preset (Qwen2.5-1.5B; label_smoothing intentionally left at 0):
weight_decay_decoder=0.1
weight_decay_other=0.01
llrd_decay=0.9
attention_dropout=0.1
max_grad_norm=1.0
# Optimizer + scheduler family shared by every stage. Each stage sets its own peak
# learning_rate, cosine restart period (T_0) and learning_rate_override below.
# CosineAnnealingWarmRestarts steps per epoch: with T_mult=2 the restart lengths
# are T_0, 2*T_0, 4*T_0, ... epochs (epochs=700). eta_min is the LR floor.
# learning_rate_override>0 rebuilds the optimizer+scheduler from scratch on resume
# (discards restored state) so a stage that continues from the previous stage's
# checkpoint starts a FRESH cosine cycle at its own peak LR. Stage 1 trains from a
# fresh mixed-decoder init, so it uses override=0 (no state to discard).
optimizer_name='AdamW'
optimizer_params='{"weight_decay": 0.01, "betas": [0.9, 0.98], "eps": 1e-8}'
learning_rate_scheduler_name='CosineAnnealingWarmRestarts'

# ---- Stage 1: warm-up (low compression ~15.75x, short target) ---------------
next_fixed_win_size=8          # N = 8 * 126 = 1008 ctx tokens
next_fixed_target_toks=256     # K
emb_exp_rate=8                 # 8 * 8 = 64 soft tokens
max_seq_len=384                # 64 + 0 + 256 = 320, headroom to 384
freeze_decoder_epochs=8        # let the soft-token bridge warm up first
learning_rate=5e-5             # highest peak: cold bridge, easiest task
learning_rate_override=0       # fresh init -> nothing to discard
learning_rate_scheduler_params='{"T_0": 30, "T_mult": 2, "eta_min": 1e-7}'


# ---- Stage 2: medium compression (~31.5x, longer target) --------------------
# pretrained_mixed_decoder_model_path=$train_root_path/mixeddecoder-20260706_213340-bertbaseuncased-d768-embEncCls-inp128-decQwen2.51.5b-msl384-dtypeBf16-sepF-pallF-pfirstT-eer8-ewn10x10-frzencF-dsNext-mnt64-srcpg_bo_ar_go_gu-trn_lr5e-05_bs8_wdD0.1_wdO0.01_llrd0.9_attdp0.1_gc1.0
# next_fixed_win_size=16         # N = 16 * 126 = 2016 ctx tokens
# next_fixed_target_toks=384     # K
# emb_exp_rate=4                 # 16 * 4 = 64 soft tokens
# max_seq_len=512                # 64 + 0 + 384 = 448, headroom to 512
# freeze_decoder_epochs=4
# learning_rate=3e-5             # lower peak: harder task, resuming warm weights
# learning_rate_override=3e-5    # rebuild optimizer+scheduler -> fresh cosine cycle
# learning_rate_scheduler_params='{"T_0": 40, "T_mult": 2, "eta_min": 1e-7}'

# ---- Stage 3: high compression (~63x, full target) --------------------------
# next_fixed_win_size=32         # N = 32 * 126 = 4032 ctx tokens
# next_fixed_target_toks=512     # K
# emb_exp_rate=2                 # 32 * 2 = 64 soft tokens
# max_seq_len=640                # 64 + 0 + 512 = 576, headroom to 640
# freeze_decoder_epochs=0
# learning_rate=2e-5             # lowest peak: hardest task, refine only
# learning_rate_override=2e-5    # rebuild optimizer+scheduler -> fresh cosine cycle
# learning_rate_scheduler_params='{"T_0": 50, "T_mult": 2, "eta_min": 1e-7}'
# =============================================================================


export PYTHONPATH=$PYTHONPATH:$mllm_src_path

# Reduce allocator fragmentation so large transient buffers (e.g. the
# cross-entropy logits over the ~152k Qwen vocab) can reuse freed blocks.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
  --prompt-first $prompt_first \
  --decoder-only $decoder_only \
  --emb-exp-rate $emb_exp_rate \
  --emb-win-min-size $emb_win_min_size \
  --emb-win-max-size $emb_win_max_size \
  --use-interactive-extractor $use_interactive_extractor \
  --ie-exp-rate $ie_exp_rate \
  --ie-num-layers $ie_num_layers \
  --ie-attn-type $ie_attn_type \
  --ie-n-heads $ie_n_heads \
  --ie-mlp-ratio $ie_mlp_ratio \
  --ie-dropout $ie_dropout \
  --ie-norm-first $ie_norm_first \
  --ie-max-ctx $ie_max_ctx \
  --ie-max-prompt-len $ie_max_prompt_len \
  --ie-prompt-in-stream $ie_prompt_in_stream \
  --train-ds-types "$train_ds_types" \
  --train-ds-batches-per-cycle "$train_ds_batches_per_cycle" \
  --train-ds-loss-weights "$train_ds_loss_weights" \
  --normalize-train-ds-loss-weights $normalize_train_ds_loss_weights \
  --min-next-toks $min_next_toks \
  --next-fixed-win-size $next_fixed_win_size \
  --next-fixed-target-toks $next_fixed_target_toks \
  --next-sources "$next_sources" \
  --next-prompt "$next_prompt" \
  --keyval-min-pairs $keyval_min_pairs \
  --keyval-max-pairs $keyval_max_pairs \
  --keyval-value-max-words $keyval_value_max_words \
  --jsonfield-min-fields $jsonfield_min_fields \
  --jsonfield-max-fields $jsonfield_max_fields \
  --jsonfield-max-depth $jsonfield_max_depth \
  --jsonfield-max-array-len $jsonfield_max_array_len \
  --jsonfield-value-max-words $jsonfield_value_max_words \
  --jsonata-min-fields $jsonata_min_fields \
  --jsonata-max-fields $jsonata_max_fields \
  --jsonata-max-depth $jsonata_max_depth \
  --jsonata-max-array-len $jsonata_max_array_len \
  --jsonata-value-max-words $jsonata_value_max_words \
  --jsonata-transform-prob $jsonata_transform_prob \
  --xmlxpath-min-nodes $xmlxpath_min_nodes \
  --xmlxpath-max-nodes $xmlxpath_max_nodes \
  --xmlxpath-max-depth $xmlxpath_max_depth \
  --xmlxpath-max-children $xmlxpath_max_children \
  --xmlxpath-value-max-words $xmlxpath_value_max_words \
  --sql-min-rows $sql_min_rows \
  --sql-max-rows $sql_max_rows \
  --sql-min-cols $sql_min_cols \
  --sql-max-cols $sql_max_cols \
  --sql-value-max-words $sql_value_max_words \
  --sql-transform-prob $sql_transform_prob \
  --structured-fill-to-budget $structured_fill_to_budget \
  --structured-fill-frac $structured_fill_frac \
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
  --freeze-decoder-epochs $freeze_decoder_epochs \
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

