
code_path=$HOME/prog
data_path=$HOME/data
wiki_ds_name=20200501.en
#wiki_ds_name=20220301.en

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fname=encmix_bert_cfg_01_base.yaml
#model_cfg_fname=encmix_bert_cfg_02_large.yaml

config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fpath=$config_dir_path/$model_cfg_fname
train_root_path=$data_path/train_mllm_encmix_bert

inp_len=128
#inp_len=256
#out_embs_type=inp
out_embs_type=new
train_ds_type=qna

#device=cpu
#epochs=5
#train_epoch_steps=20
#val_epoch_steps=20
#batch_size=5

device=cuda
epochs=700
train_epoch_steps=500
val_epoch_steps=50
batch_size=15
#train_subdir=last

learning_rate=0.0001
learning_rate=0.00005
random_seed=200

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_07_01_train_encmix_bert.py \
  --data-path $data_path \
  --wiki-ds-name $wiki_ds_name \
  --train-ds-type $train_ds_type \
  --train-root-path $train_root_path \
  --train-subdir "$train_subdir" \
  --model-cfg-fpath $model_cfg_fpath \
  --inp-len $inp_len \
  --out-embs-type $out_embs_type \
  --batch-size $batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --random-seed $random_seed
#"

