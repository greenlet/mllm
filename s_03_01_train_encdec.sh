
code_path=$HOME/prog
data_path=$HOME/data
wiki_data_path=$data_path/wiki_20200501_en
ds_subdir=ch_100_fixed
wiki_ds_path=$wiki_data_path/$ds_subdir

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
tokenizer_cfg_fname=tokenizer_cfg_01.yaml
model_cfg_fname=encdec_model_cfg_01.yaml
model_level=0
n_enc_layers=3
n_dec_layers=3
dec_with_vocab_decoder=true


tokenizer_cfg_fpath=$config_dir_path/$tokenizer_cfg_fname
model_cfg_fpath=$config_dir_path/$model_cfg_fname
train_root_path=$data_path/train_mllm_encdec_${model_level}

# device=cpu
# epochs=5
# train_epoch_steps=20
# val_epoch_steps=20
# docs_batch_size=10
# max_chunks_per_doc=3

device=cuda
epochs=500
train_epoch_steps=500
val_epoch_steps=50
docs_batch_size=7
max_chunks_per_doc=3
#train_subdir=last

learning_rate=0.0001

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
# echo "
python s_03_01_train_encdec.py \
  --ds-dir-path $wiki_ds_path \
  --train-root-path $train_root_path \
  --train-subdir "$train_subdir" \
  --tokenizer-cfg-fpath $tokenizer_cfg_fpath \
  --model-cfg-fpath $model_cfg_fpath \
  --model-level $model_level \
  --n-enc-layers $n_enc_layers \
  --n-dec-layers $n_dec_layers \
  --dec-with-vocab-decoder $dec_with_vocab_decoder \
  --docs-batch-size $docs_batch_size \
  --max-chunks-per-doc $max_chunks_per_doc \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps
# "


