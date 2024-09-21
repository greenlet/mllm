
code_path=$HOME/prog
data_path=$HOME/data
ds_dir_path=$data_path/ranker_embs_msmarco_fever
msmarco_data_path=$data_path/msmarco
fever_data_path=$data_path/fever

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
encdec_model_cfg_fname=encdec_model_cfg_02.yaml
ranker_model_cfg_fname=encdec_model_cfg_02.yaml
encdec_model_cfg_fpath=$config_dir_path/$encdec_model_cfg_fname
ranker_model_cfg_fpath=$config_dir_path/$ranker_model_cfg_fname
model_level=1

train_ranker_root_path=$data_path/train_mllm_ranker_qrels_1
train_encdec_root_path=$data_path/train_mllm_encdec_1
encdec_pretrained_model_path=$train_encdec_root_path/encdec-l1-20240918_063547-msmarco-fever

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
python s_05_train_02_ranker_qrels_embs.py \
  --ds-dir-path $msmarco_data_path \
  --ds-dir-paths $msmarco_data_path $fever_data_path \
  --train-root-path $train_ranker_root_path \
  --train-subdir "$train_subdir" \
  --ranker-model-cfg-fpath $ranker_model_cfg_fpath \
  --model-level $model_level \
  --chunks-batch-size $chunks_batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --encdec-model-cfg-fpath $encdec_model_cfg_fpath \
  --encdec-pretrained-model-path $encdec_pretrained_model_path
#"


