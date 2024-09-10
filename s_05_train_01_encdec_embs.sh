
code_path=$HOME/prog
data_path=$HOME/data
wiki_data_path=$data_path/wiki_20200501_en
ds_subdir=ch_100_fixed
wiki_ds_path=$wiki_data_path/$ds_subdir

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg_v001
tokenizer_cfg_fname=tokenizer_cfg_01.yaml
model_cfg_fname=ranker_model_cfg_02.yaml
model_level=1

tokenizer_cfg_fpath=$config_dir_path/$tokenizer_cfg_fname
model_cfg_fpath=$config_dir_path/$model_cfg_fname
train_root_path=$data_path/train_mllm_encdec_${model_level}

device=cpu
epochs=5
train_epoch_steps=20
val_epoch_steps=20
docs_batch_size=3
max_chunks_per_doc=2

#device=cuda
#epochs=500
#train_epoch_steps=500
#val_epoch_steps=50
#docs_batch_size=10
#max_chunks_per_doc=3
#train_subdir=last

learning_rate=0.0001

mllm_src_path=$code_path/mllm
export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_05_train_01_encdec_embs.py \
  --ds-dir-path $wiki_ds_path \
  --train-root-path $train_root_path \
  --train-subdir "$train_subdir" \
  --tokenizer-cfg-fpath $tokenizer_cfg_fpath \
  --model-cfg-fpath $model_cfg_fpath \
  --model-levl $model_level \
  --docs-batch-size $docs_batch_size \
  --max-chunks-per-doc $max_chunks_per_doc \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps
#"


