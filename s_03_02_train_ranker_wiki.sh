
code_path=$HOME/prog
data_path=$HOME/data
wiki_data_path=$data_path/wiki_20200501_en
#ds_subdir=ch_100_nonfixed
ds_subdir=ch_100_fixed
wiki_ds_path=$wiki_data_path/$ds_subdir
train_ranker_root_path=$data_path/train_mllm_ranker_wiki
train_encdec_root_path=$data_path/train_mllm_encdec

device=cpu
#epochs=5
#train_epoch_steps=20
#val_epoch_steps=20

device=cuda
epochs=5
train_epoch_steps=500
val_epoch_steps=50
docs_batch_size=3
max_chunks_per_doc=3
pretrained_model_path=$train_encdec_root_path/encdec-20240718_221554-wiki_20200501_en-ch_100_fixed


device=cuda
epochs=200
train_epoch_steps=5000
val_epoch_steps=500
docs_batch_size=20
max_chunks_per_doc=5
pretrained_model_path=$train_encdec_root_path/encdec-20240718_221554-wiki_20200501_en-ch_100_fixed
#train_subdir=last

learning_rate=0.0001

mllm_src_path=$code_path/mllm
export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_03_02_train_ranker_wiki.py \
  --ds-dir-path $wiki_ds_path \
  --train-root-path $train_ranker_root_path \
  --train-subdir "$train_subdir" \
  --docs-batch-size $docs_batch_size \
  --max-chunks-per-doc $max_chunks_per_doc \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --pretrained-model-path $pretrained_model_path
#"


