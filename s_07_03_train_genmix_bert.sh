
code_path=$HOME/prog
data_path=$HOME/data

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fname=genmix_bert_cfg_01_base.yaml

config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fpath=$config_dir_path/$model_cfg_fname
train_root_path=$data_path/train_mllm_genmix_bert
train_encdec_root_path=$data_path/train_mllm_encdec_bert

bert_model_name=bert-base-uncased
#bert_model_name=bert-large-uncased
inp_len=128
#inp_len=256
#n_first_embs=1
#n_second_embs=1
#emb_agg_type=mat
#emb_exp_type=non
n_first_embs=1
n_second_embs=128
#emb_agg_type=fst
emb_exp_type=mat
emb_exp_type=mtb
#train_ds_type=qna
#train_ds_type=sum
train_ds_type=wki
#mask_tgt=false
mask_tgt=true
#max_tgt_len_freq=0.2
#max_tgt_len=10
#max_inp_chunks=1
#max_out_toks=50

#max_inp_chunks=10
#n_first_embs=5
#n_second_embs=5
#emb_agg_type=fst
#emb_exp_type=non

device=cpu
epochs=5
train_epoch_steps=20
val_epoch_steps=20
batch_size=5

#pretrained_model_path=$train_encdec_root_path/encdecbert-20250131_223521-bert-base-uncased-d768-emb_cls-inp128-lrs7x1-enh_mmbb-step2-h12-dp0-t0.0
device=cuda
epochs=700
train_epoch_steps=500
val_epoch_steps=50
#batch_size=15
batch_size=1
#train_subdir=last

#learning_rate=0.0001
#learning_rate=0.00005
learning_rate=0.00001
random_seed=200

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_07_03_train_genmix_bert.py \
  --data-path $data_path \
  --train-root-path $train_root_path \
  --pretrained-model-path "$pretrained_model_path" \
  --train-subdir "$train_subdir" \
  --train-ds-type "$train_ds_type" \
  --mask-tgt "$mask_tgt" \
  --max-tgt-len-freq $max_tgt_len_freq \
  --max-tgt-len $max_tgt_len \
  --model-cfg-fpath $model_cfg_fpath \
  --bert-model-name $bert_model_name \
  --inp-len $inp_len \
  --n-first-embs $n_first_embs \
  --n-second-embs $n_second_embs \
  --emb-agg-type $emb_agg_type \
  --emb-exp-type $emb_exp_type \
  --max-inp-chunks $max_inp_chunks \
  --max-out-toks $max_out_toks \
  --batch-size $batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --random-seed $random_seed
#"

