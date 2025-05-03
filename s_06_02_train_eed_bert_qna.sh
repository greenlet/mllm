
code_path=$HOME/prog
data_path=$HOME/data
train_root_path=$data_path/train_mllm_eed_bert_qna
train_encdec_root_path=$data_path/train_mllm_encdec_bert

mllm_src_path=$code_path/mllm

inp_len=128

device=cpu
epochs=5
train_epoch_steps=20
val_epoch_steps=20
batch_size=3

device=cuda
epochs=500
train_epoch_steps=500
val_epoch_steps=50
batch_size=10
#batch_size=5

#pretrained_model_path=$train_encdec_root_path/encdecbert-20250131_223521-bert-base-uncased-d768-emb_cls-inp128-lrs7x1-enh_mmbb-step2-h12-dp0-t0.0
#train_subdir=last
#in_empty_ans=true
in_empty_ans=false
ques_inp=enc
#ques_inp=dec
enc_emb_exp_type=emb
#enc_emb_exp_type=mat
enc_emb_exp_bias=true


#learning_rate=0.0001
learning_rate=0.00002
random_seed=111

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_06_02_train_eed_bert_qna.py \
  --train-root-path $train_root_path \
  --pretrained-model-path "$pretrained_model_path" \
  --train-subdir "$train_subdir" \
  --inp-len $inp_len \
  --batch-size $batch_size \
  --in-empty-ans $in_empty_ans \
  --ques-inp $ques_inp \
  --enc-emb-exp-type $enc_emb_exp_type \
  --enc-emb-exp-bias $enc_emb_exp_bias \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --random-seed $random_seed \
#"

