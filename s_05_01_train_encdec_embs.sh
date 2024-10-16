
code_path=$HOME/prog
data_path=$HOME/data
ds_dir_path=$data_path/ranker_embs_msmarco_fever
msmarco_data_path=$data_path/msmarco
fever_data_path=$data_path/fever

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
# model_cfg_fname=encdec_model_cfg_02.yaml
model_cfg_fname=encdec_model_cfg_03.yaml
model_level=1

model_cfg_fpath=$config_dir_path/$model_cfg_fname
train_root_path=$data_path/train_mllm_encdec_${model_level}
#train_subdir=encdec-l1-20241005_175446-msmarco-fever

#device=cpu
#epochs=5
#train_epoch_steps=20
#val_epoch_steps=20
#docs_batch_size=100

device=cuda
epochs=300
train_epoch_steps=500
val_epoch_steps=50
docs_batch_size=2000
#train_subdir=last

learning_rate=0.0001

mllm_src_path=$code_path/mllm
export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_05_01_train_encdec_embs.py \
  --ds-dir-path $ds_dir_path \
  --ds-dir-paths $msmarco_data_path $fever_data_path \
  --train-root-path $train_root_path \
  --train-subdir "$train_subdir" \
  --model-cfg-fpath $model_cfg_fpath \
  --model-level $model_level \
  --docs-batch-size $docs_batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps
#"


