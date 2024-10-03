
code_path=$HOME/prog
data_path=$HOME/data
model_level=1
ds_dir_path=$data_path/encdec_embs_${model_level}_msmarco_fever

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
ranker_model_cfg_fname=encdec_model_cfg_02.yaml
ranker_model_cfg_fpath=$config_dir_path/$ranker_model_cfg_fname

train_ranker_root_path=$data_path/train_mllm_ranker_qrels_${model_level}
train_subdir=""

device=cpu
epochs=5
train_epoch_steps=20
val_epoch_steps=20
chunks_batch_size=10

#device=cuda
#epochs=5
#train_epoch_steps=500
#val_epoch_steps=50
#chunks_batch_size=3

learning_rate=0.0001

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_05_03_train_ranker_qrels_embs.py \
  --ds-dir-path $ds_dir_path \
  --train-root-path $train_ranker_root_path \
  --train-subdir "$train_subdir" \
  --ranker-model-cfg-fpath $ranker_model_cfg_fpath \
  --model-level $model_level \
  --chunks-batch-size $chunks_batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps
#"


