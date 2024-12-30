
code_path=$HOME/prog
data_path=$HOME/data
msmarco_data_path=$data_path/msmarco
fever_data_path=$data_path/fever
train_ranker_root_path=$data_path/train_mllm_ranker_hg_qrels
train_encdec_root_path=$data_path/train_mllm_encdec_hg

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
tokenizer_cfg_fname=tokenizer_cfg_01.yaml
model_cfg_fname=ranker_hg_cfg_01.yaml
model_cfg_fname=ranker_hg_cfg_02.yaml

tokenizer_cfg_fpath=$config_dir_path/$tokenizer_cfg_fname
model_cfg_fpath=$config_dir_path/$model_cfg_fname
train_root_path=$data_path/train_mllm_ranker_hg

inp_len=128
n_similar_layers=1
reduct_type=avg
enhance_type=mmbeg
#enhance_type=mmbb
#pos_enc_type=num
pos_enc_type=emb
dec_dropout_rate=-1
#dec_with_bias=false
#dec_with_bias=true
dec_mlp_sizes=-1

device=cpu
epochs=5
train_epoch_steps=20
val_epoch_steps=20
docs_batch_size=3
pretrained_model_path=$train_encdec_root_path/encdechg-20241216_224415-inp128-pos_emb-lrs7x1-rdc_avg-enh_mmbeg-step2-d512-h8

device=cuda
epochs=500
train_epoch_steps=500
val_epoch_steps=50
docs_batch_size=30
pretrained_model_path=$train_encdec_root_path/encdechg-20241216_224415-inp128-pos_emb-lrs7x1-rdc_avg-enh_mmbeg-step2-d512-h8
#train_subdir=last

learning_rate=0.0001

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_03_06_train_ranker_hg_qrels.py \
  --ds-dir-paths $msmarco_data_path $fever_data_path \
  --train-root-path $train_ranker_root_path \
  --train-subdir "$train_subdir" \
  --tokenizer-cfg-fpath $tokenizer_cfg_fpath \
  --model-cfg-fpath $model_cfg_fpath \
  --inp-len $inp_len \
  --n-similar-layers $n_similar_layers \
  --reduct-type $reduct_type \
  --enhance-type $enhance_type \
  --pos-enc-type $pos_enc_type \
  --dec-dropout-rate $dec_dropout_rate \
  --dec-with-bias $dec_with_bias \
  --dec-mlp-sizes $dec_mlp_sizes \
  --docs-batch-size $docs_batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --pretrained-model-path $pretrained_model_path
#"

