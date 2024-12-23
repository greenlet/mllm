
code_path=$HOME/prog
data_path=$HOME/data
wiki_ds_name=20200501.en

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
tokenizer_cfg_fname=tokenizer_cfg_01.yaml
#model_cfg_fname=encdec_hg_cfg_01.yaml
#model_cfg_fname=encdec_hg_cfg_02.yaml
model_cfg_fname=encdec_hg_cfg_03.yaml

config_dir_path=$mllm_src_path/mllm/config/cfg
tokenizer_cfg_fpath=$config_dir_path/$tokenizer_cfg_fname
model_cfg_fpath=$config_dir_path/$model_cfg_fname
train_root_path=$data_path/train_mllm_encdec_hg

#inp_len=256
inp_len=128
inp_len=256
n_similar_layers=1
#n_similar_layers=2
#reduct_type=matmul
#reduct_type=decim
reduct_type=avg

#enhance_type=matmul
enhance_type=mmbeg
#enhance_type=mmbb
#pos_enc_type=num
pos_enc_type=emb

#device=cpu
#epochs=5
#train_epoch_steps=20
#val_epoch_steps=20
#docs_batch_size=10

device=cuda
epochs=700
train_epoch_steps=500
val_epoch_steps=50
#docs_batch_size=20
docs_batch_size=10
#train_subdir=last

learning_rate=0.0001
#learning_rate=0.001

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_03_05_train_encdec_hg.py \
  --data-path $data_path \
  --wiki-ds-name $wiki_ds_name \
  --train-root-path $train_root_path \
  --train-subdir "$train_subdir" \
  --tokenizer-cfg-fpath $tokenizer_cfg_fpath \
  --model-cfg-fpath $model_cfg_fpath \
  --inp-len $inp_len \
  --n-similar-layers $n_similar_layers \
  --reduct-type $reduct_type \
  --enhance-type $enhance_type \
  --pos-enc-type $pos_enc_type \
  --docs-batch-size $docs_batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps
#"

