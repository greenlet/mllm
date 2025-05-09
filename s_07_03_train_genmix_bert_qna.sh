
code_path=$HOME/prog
data_path=$HOME/data

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fname=genmix_bert_cfg_01_base_tte.yaml

config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fpath=$config_dir_path/$model_cfg_fname
train_root_path=$data_path/train_mllm_genmix_bert
train_encdec_root_path=$data_path/train_mllm_encdec_bert

inp_len=128
#inp_len=256

#device=cpu
#epochs=5
#train_epoch_steps=20
#val_epoch_steps=20
#batch_size=5

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
python s_07_03_train_genmix_bert_qna.py \
  --data-path $data_path \
  --train-root-path $train_root_path \
  --pretrained-model-path "$pretrained_model_path" \
  --train-subdir "$train_subdir" \
  --model-cfg-fpath $model_cfg_fpath \
  --inp-len $inp_len \
  --batch-size $batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --random-seed $random_seed
#"

