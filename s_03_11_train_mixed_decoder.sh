
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

# Decoder type: gpt2 or bert_dec
# decoder_type=gpt2
# decoder_model_name=gpt2
decoder_type=bertdec
decoder_model_name=bert-base-uncased

train_ds_type=cite
train_ds_type=qna
train_ds_type=next

min_next_toks=64

max_seq_len=384
freeze_encoder=false
use_sep=true
prompt_all=false
emb_exp_rate=4
emb_win_min_size=2
emb_win_max_size=10

mask_tokens=false
mask_sep_freq=0.5
mask_sep_frac=0.15
mask_seq_freq=0.5
mask_seq_max_frac=0.2
mask_seq_max_len=20
mask_n_last_toks=0

pretrained_encdec_model_path=$train_root_path/encdecbert-20260110_193915-bertbaseuncased-d768-embCls-inp128-lrs7x1-enhMmbb-step2-h12-dp0-t0.0
# pretrained_mixed_decoder_model_path=$train_root_path/mixeddecoder-20260304_105309-pre_encdecbert20260110193915-bertbaseuncased-d768-embEncCls-inp128-decGpt2-decmgpt2-msl384-sepT-pallF-eer4-ewn10x10-frzencF-trn_lr5e-05_bs30
pretrained_mixed_decoder_model_path=$train_root_path/mixeddecoder-20260316_221645-pre_mixeddecoder20260304105309-bertbaseuncased-d768-embEncCls-inp128-decBertbaseuncased-msl384-sepT-pallF-eer4-ewn10x10-frzencF-dsCite-trn_lr5e-05_bs40
pretrained_mixed_decoder_model_path=$train_root_path/mixeddecoder-20260319_130017-pre_mixeddecoder20260316221645-bertbaseuncased-d768-embEncCls-inp128-decBertbaseuncased-msl384-sepT-pallF-eer4-ewn10x10-frzencF-dsCite-msk_sep0.5x0.15_seq0.5x0.2x20_last0-trn_lr5e-05_bs40
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
# docs_batch_size=20
docs_batch_size=15
world_size=4


learning_rate=0.00005
random_seed=200

optimizer_name='AdamW'
optimizer_params='{}'
learning_rate_scheduler_name='ReduceLROnPlateau'
learning_rate_scheduler_params='{"mode": "min", "factor": 0.5, "patience": 10, "threshold": 1e-6, "min_lr": 1e-8}'

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

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
  --max-seq-len $max_seq_len \
  --freeze-encoder $freeze_encoder \
  --use-sep $use_sep \
  --prompt-all $prompt_all \
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
  --optimizer-name $optimizer_name \
  --optimizer-params "$optimizer_params" \
  --learning-rate-scheduler-name $learning_rate_scheduler_name \
  --learning-rate-scheduler-params "$learning_rate_scheduler_params" \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --random-seed $random_seed \
  --pretrained-encdec-model-path "$pretrained_encdec_model_path" \
  --pretrained-mixed-decoder-model-path "$pretrained_mixed_decoder_model_path" \
  --world-size $world_size

