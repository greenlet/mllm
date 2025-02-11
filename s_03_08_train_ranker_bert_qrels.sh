
code_path=$HOME/prog
data_path=$HOME/data
msmarco_data_path=$data_path/msmarco
fever_data_path=$data_path/fever
train_ranker_root_path=$data_path/train_mllm_ranker_bert_qrels
train_encdec_root_path=$data_path/train_mllm_encdec_bert

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fpath=$config_dir_path/$model_cfg_fname
model_cfg_fname=ranker_bert_cfg_01.yaml

model_cfg_fpath=$config_dir_path/$model_cfg_fname

inp_len=128
bert_emb_type=cls
dec_mlp_layers="768b"
#dec_mlp_layers="2048b,tanh,768b"
dec_mlp_layers=""
#train_dec_only=true
train_dec_only=false

device=cpu
epochs=5
train_epoch_steps=20
val_epoch_steps=20
docs_batch_size=3
#pretrained_model_path=$train_encdec_root_path/encdecbert-20250131_223521-bert-base-uncased-d768-emb_cls-inp128-lrs7x1-enh_mmbb-step2-h12-dp0-t0.0

device=cuda
epochs=500
train_epoch_steps=500
val_epoch_steps=50
#docs_batch_size=30
#docs_batch_size=15
docs_batch_size=7
#pretrained_model_path=$train_encdec_root_path/encdecbert-20250131_223521-bert-base-uncased-d768-emb_cls-inp128-lrs7x1-enh_mmbb-step2-h12-dp0-t0.0
#pretrained_model_path=
#train_subdir=last

#loss_type=avg
#loss_type=max
loss_type=lft

learning_rate=0.0001
learning_rate=0.00005
random_seed=111

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_03_08_train_ranker_bert_qrels.py \
  --ds-dir-paths $msmarco_data_path $fever_data_path \
  --train-root-path $train_ranker_root_path \
  --train-subdir "$train_subdir" \
  --model-cfg-fpath $model_cfg_fpath \
  --inp-len $inp_len \
  --bert-emb-type $bert_emb_type \
  --dec-mlp-layers "$dec_mlp_layers" \
  --docs-batch-size $docs_batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --pretrained-model-path "$pretrained_model_path" \
  --train-dec-only $train_dec_only \
  --random-seed $random_seed \
  --loss-type $loss_type
#"

