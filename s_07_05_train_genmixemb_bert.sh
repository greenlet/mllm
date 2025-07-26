
code_path=$HOME/prog
data_path=$HOME/data

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fname=genmixemb_bert_cfg_01_base.yaml

config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fpath=$config_dir_path/$model_cfg_fname
train_root_path=$data_path/train_mllm_genmixemb_bert
train_root_path=$data_path/train_mllm_genmixembbert_qna
train_encdec_root_path=$data_path/train_mllm_encdec_bert

#train_ds_type=wki
train_ds_type=qna
#train_ds_type=sum

bert_model_name=bert-base-uncased
#bert_model_name=bert-large-uncased
max_out_toks=50
toks_agg_type=brt
toks_agg_type=pyr
#bert_agg_n_subseq_toks=0
bert_agg_n_subseq_toks=2
#bert_agg_n_subseq_toks=8
pyr_agg_type=decim
#pyr_agg_type=matmul
#pyr_agg_type=avg
#pyr_agg_type=sub
pyr_agg_step=2
pyr_agg_n_levels=3
pyr_agg_n_layers_per_level=2
#pyr_agg_n_layers_per_level=3
pyr_agg_step=0
train_agg_model=false
#train_agg_model=true
pred_next_sent=true

n_toks_min=20
n_toks_max=100
n_toks_max=256
n_toks_max=384
#n_toks_max=512
mask_tokens=false
#mask_tokens=true
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
#pretrained_model_path=$train_root_path/genmixemb-20250713_202718-bertbaseuncased-d768-mxo50-aggBrt-sub0-dsWki-tmax100-tragF
#pretrained_model_path=$train_root_path/genmixemb-20250721_083250-bertbaseuncased-d768-mxo50-aggPyr-agtDecim-stp0-lvl1-lrs2-dsWki-tmax256-tragF-nxtsnt
#pretrained_model_path=$train_root_path/genmixemb-20250721_212402-bertbaseuncased-d768-mxo50-aggPyr-agtDecim-stp2-lvl2-lrs2-dsWki-tmax512-tragT-nxtsnt

device=cuda
epochs=700
train_epoch_steps=500
val_epoch_steps=50
#batch_size=20
#batch_size=15
batch_size=10
#batch_size=5
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
  --pyr-agg-type $pyr_agg_type \
  --pyr-agg-step $pyr_agg_step \
  --pyr-agg-n-levels $pyr_agg_n_levels \
  --pyr-agg-n-layers-per-level $pyr_agg_n_layers_per_level \
  --train-agg-model $train_agg_model \
  --pred-next-sent $pred_next_sent \
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

