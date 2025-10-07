
code_path=$HOME/prog
data_path=$HOME/data
#wiki_ds_name=20200501.en
wiki_ds_name=20220301.en

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fname=encdec_bert_cfg_01.yaml

config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fpath=$config_dir_path/$model_cfg_fname
train_root_path=$data_path/train_mllm_encdec_bert

bert_model_name=bert-base-uncased
bert_emb_type=cls
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
mask_tokens=false
mask_sep_freq=0.5
mask_sep_frac=0.15
mask_seq_freq=0.5
mask_seq_max_frac=0.2
mask_seq_max_len=20
enforce_encoder_mask_understanding=false

bert_model_name=bert-base-uncased
bert_emb_type=cls
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
mask_tokens=true
mask_sep_freq=0.5
mask_sep_frac=0.04
mask_seq_freq=0.5
mask_seq_max_frac=0.05
mask_seq_max_len=5
next_tok_pred=true
enforce_encoder_mask_understanding=true

#bert_model_name=bert-base-uncased
#bert_emb_type=cls
##inp_len=384
#inp_len=256
#dec_n_layers=0
#dec_n_similar_layers=1
##dec_enhance_type=matmul
##dec_enhance_type=mmbeg
#dec_enhance_type=mmbb
#dec_dropout_rate=0
#mask_tokens=false
#mask_sep_freq=0.5
#mask_sep_frac=0.15
#mask_seq_freq=0.5
#mask_seq_max_frac=0.2
#mask_seq_max_len=20
#enforce_encoder_mask_understanding=false



#pretrained_model_path=$train_root_path/encdecbert-20250131_223521-bert-base-uncased-d768-emb_cls-inp128-lrs7x1-enh_mmbb-step2-h12-dp0-t0.0
#pretrained_model_path=$train_root_path/encdecbert-20251004_224422-bertbaseuncased-d768-embCls-inp128-lrs7x1-enhMmbb-step2-h12-dp0-t0.0


device=cpu
epochs=5
train_epoch_steps=20
val_epoch_steps=20
docs_batch_size=5

device=cuda
epochs=700
train_epoch_steps=500
val_epoch_steps=50
#docs_batch_size=25
# docs_batch_size=20
docs_batch_size=15
# docs_batch_size=10
# docs_batch_size=3
#train_subdir=last


learning_rate=0.0001
learning_rate=0.00005
#learning_rate=0.00001
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
  --bert-model-name $bert_model_name \
  --bert-emb-type $bert_emb_type \
  --inp-len $inp_len \
  --dec-enhance-type $dec_enhance_type \
  --dec-n-layers $dec_n_layers \
  --dec-n-similar-layers $dec_n_similar_layers \
  --mask-tokens $mask_tokens \
  --mask-sep-freq $mask_sep_freq \
  --mask-sep-frac $mask_sep_frac \
  --mask-seq-freq $mask_seq_freq \
  --mask-seq-max-frac $mask_seq_max_frac \
  --mask-seq-max-len $mask_seq_max_len \
  --next-tok-pred $next_tok_pred \
  --dec-dropout-rate $dec_dropout_rate \
  --docs-batch-size $docs_batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --random-seed $random_seed \
  --pretrained-model-path "$pretrained_model_path" \
  --enforce-encoder-mask-understanding $enforce_encoder_mask_understanding
#"

