
code_path=$HOME/prog
data_path=$HOME/data
wiki_ds_name=20200501.en
#wiki_ds_name=20220301.en

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fname=encdec_bert_cfg_01.yaml

config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fpath=$config_dir_path/$model_cfg_fname
train_root_path=$data_path/train_mllm_encdec_bert

inp_len=128
#inp_len=256
#dec_n_layers=7
dec_n_layers=0

dec_n_similar_layers=1
#n_similar_layers=2

#dec_enhance_type=matmul
#dec_enhance_type=mmbeg
dec_enhance_type=mmbb
dec_dropout_rate=0

device=cpu
epochs=5
train_epoch_steps=20
val_epoch_steps=20
docs_batch_size=10

# device=cuda
# epochs=700
# train_epoch_steps=500
# val_epoch_steps=50
# docs_batch_size=30
# #docs_batch_size=7
# #docs_batch_size=15
# #train_subdir=last
# pretrained_model_path=$train_root_path/encdechg-20241216_224415-inp128-pos_emb-lrs7x1-rdc_avg-enh_mmbeg-step2-d512-h8-t1

learning_rate=0.0001
# learning_rate=0.00005
random_seed=200

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_03_07_train_encdec_bert.py \
  --data-path $data_path \
  --wiki-ds-name $wiki_ds_name \
  --train-root-path $train_root_path \
  --train-subdir "$train_subdir" \
  --model-cfg-fpath $model_cfg_fpath \
  --inp-len $inp_len \
  --dec-enhance-type $enhance_type \
  --dec-n-layers $dec_n_layers \
  --dec-n-similar-layers $dec_n_similar_layers \
  --dec-dropout-rate $dropout_rate \
  --docs-batch-size $docs_batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --random-seed $random_seed \
  --pretrained-model-path $pretrained_model_path
#"
