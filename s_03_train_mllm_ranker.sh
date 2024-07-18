#!/bin/zsh

code_path=$HOME/prog
data_path=$HOME/data
wiki_data_path=$data_path/wiki_20200501_en
#ds_subdir=ch_100_nonfixed
ds_subdir=ch_100_fixed
wiki_ds_path=$wiki_data_path/$ds_subdir
train_ranker_root_path=$data_path/train_mllm_ranker
train_encdec_root_path=$data_path/train_mllm_encdec

device=cpu
#epochs=5
#train_epoch_steps=20
#val_epoch_steps=20

device=cuda
epochs=20
train_epoch_steps=1000
val_epoch_steps=100
docs_batch_size=5
max_chunks_per_doc=3
pretrained_model_path=$train_encdec_root_path/encdec-20240718_221554-wiki_20200501_en-ch_100_fixed

learning_rate=0.001

mllm_src_path=$code_path/mllm
export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_03_train_mllm_ranker.py \
  --ds-dir-path $wiki_ds_path \
  --train-root-path $train_ranker_root_path \
  --docs-batch-size $docs_batch_size \
  --max-chunks-per-doc $max_chunks_per_doc \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --pretrained-model-path $pretrained_model_path
#"


