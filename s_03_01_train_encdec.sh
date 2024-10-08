
code_path=$HOME/prog
data_path=$HOME/data
wiki_data_path=$data_path/wiki_20200501_en
ds_subdir=ch_100_fixed
wiki_ds_path=$wiki_data_path/$ds_subdir
train_root_path=$data_path/train_mllm_encdec

# device=cpu
# epochs=5
# train_epoch_steps=20
# val_epoch_steps=20

device=cuda
epochs=500
train_epoch_steps=500
val_epoch_steps=50
docs_batch_size=10
max_chunks_per_doc=3
train_subdir=last
# train_subdir=encdec-20240808_222352-wiki_20200501_en-ch_100_fixed

learning_rate=0.0001

mllm_src_path=$code_path/mllm
export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_03_01_train_encdec.py \
  --ds-dir-path $wiki_ds_path \
  --train-root-path $train_root_path \
  --train-subdir "$train_subdir" \
  --docs-batch-size $docs_batch_size \
  --max-chunks-per-doc $max_chunks_per_doc \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps
#"


