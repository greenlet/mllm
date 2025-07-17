
code_path=$HOME/prog
data_path=$HOME/data

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fname=genmixemb_bert_cfg_01_base.yaml

config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fpath=$config_dir_path/$model_cfg_fname
train_root_path=$data_path/train_mllm_genmixemb_bert
train_encdec_root_path=$data_path/train_mllm_encdec_bert

#train_ds_type=qna
#train_ds_type=sum
train_ds_type=wki

bert_model_name=bert-base-uncased
#bert_model_name=bert-large-uncased
max_out_toks=50
toks_agg_type=brt
#toks_agg_type=pyr
bert_agg_n_subseq_toks=0
#bert_agg_n_subseq_toks=2
pyr_agg_n_levels=0
pyr_agg_n_layers_per_level=0
train_agg_model=false
#train_agg_model=true

n_toks_min=20
n_toks_max=100
#mask_tokens=false
mask_tokens=true
mask_sep_freq=0.5
mask_sep_frac=0.15
mask_seq_freq=0.5
mask_seq_max_frac=0.2
mask_seq_max_len=20

device=cpu
epochs=5
train_epoch_steps=20
val_epoch_steps=20
batch_size=5

#pretrained_model_path=$train_encdec_root_path/encdecbert-20250131_223521-bert-base-uncased-d768-emb_cls-inp128-lrs7x1-enh_mmbb-step2-h12-dp0-t0.0
#pretrained_model_path=$train_root_path/
device=cuda
epochs=700
train_epoch_steps=500
val_epoch_steps=50
#batch_size=15
batch_size=5
#train_subdir=last

#learning_rate=0.0001
learning_rate=0.00005
#learning_rate=0.00001
random_seed=200

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_07_05_train_genmixemb_bert.py \
  --data-path $data_path \
  --train-root-path $train_root_path \
  --pretrained-model-path "$pretrained_model_path" \
  --train-subdir "$train_subdir" \
  --train-ds-type "$train_ds_type" \
  --model-cfg-fpath $model_cfg_fpath \
  --bert-model-name $bert_model_name \
  --max-out-toks $max_out_toks \
  --toks-agg-type $toks_agg_type \
  --bert-agg-n-subseq-toks $bert_agg_n_subseq_toks \
  --pyr-agg-n-levels $pyr_agg_n_levels \
  --pyr-agg-n-layers-per-level $pyr_agg_n_layers_per_level \
  --train-agg-model $train_agg_model \
  --n-toks-min $n_toks_min \
  --n-toks-max $n_toks_max \
  --mask-tokens $mask_tokens \
  --mask-sep-freq $mask_sep_freq \
  --mask-sep-frac $mask_sep_frac \
  --mask-seq-freq $mask_seq_freq \
  --mask-seq-max-frac $mask_seq_max_frac \
  --mask-seq-max-len $mask_seq_max_len \
  --batch-size $batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --random-seed $random_seed
#"

