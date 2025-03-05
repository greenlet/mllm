
code_path=$HOME/prog
data_path=$HOME/data
train_root_path=$data_path/train_mllm_eed_bert_qna

mllm_src_path=$code_path/mllm

inp_len=128

#device=cpu
#epochs=5
#train_epoch_steps=20
#val_epoch_steps=20
#batch_size=3

device=cuda
epochs=500
train_epoch_steps=500
val_epoch_steps=50
#batch_size=10
batch_size=5
#train_subdir=last

#learning_rate=0.0001
learning_rate=0.00002
random_seed=111

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_06_01_train_eed_bert_qna.py \
  --train-root-path $train_root_path \
  --train-subdir "$train_subdir" \
  --inp-len $inp_len \
  --batch-size $batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --random-seed $random_seed \
#"

