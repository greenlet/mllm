
code_path=$HOME/prog
data_path=$HOME/data

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fname=at2_cfg_01_base_tte.yaml

config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fpath=$config_dir_path/$model_cfg_fname
train_root_path=$data_path/train_mllm_genat2_bert

inp_len=128
#inp_len=256
#train_ds_type=qna
train_ds_type=sum
max_inp_chunks=10
max_out_toks=50
bert_model_name=bert-base-uncased
#bert_model_name=bert-large-uncased
enc_at2_enabled=true
dec_at2_enabled=true
last_dec_to_all_enc_at2_enabled=true

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
python s_07_04_train_genat2_bert.py \
  --data-path $data_path \
  --train-root-path $train_root_path \
  --pretrained-model-path "$pretrained_model_path" \
  --train-subdir "$train_subdir" \
  --train-ds-type "$train_ds_type" \
  --model-cfg-fpath $model_cfg_fpath \
  --bert-model-name $bert_model_name \
  --inp-len $inp_len \
  --max-inp-chunks $max_inp_chunks \
  --max-out-toks $max_out_toks \
  --enc-at2-enabled $enc_at2_enabled \
  --dec-at2-enabled $dec_at2_enabled \
  --last-dec-to-all-enc-at2-enabled $last_dec_to_all_enc_at2_enabled \
  --batch-size $batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --random-seed $random_seed
#"

