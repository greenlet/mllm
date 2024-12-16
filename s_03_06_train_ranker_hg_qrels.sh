
code_path=$HOME/prog
data_path=$HOME/data
msmarco_data_path=$data_path/msmarco
fever_data_path=$data_path/fever
train_ranker_root_path=$data_path/train_mllm_ranker_hg_qrels
train_encdec_root_path=$data_path/train_mllm_encdec_hg

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
tokenizer_cfg_fname=tokenizer_cfg_01.yaml
ranker_model_cfg_fname=ranker_hg_cfg_01.yaml

inp_len=128
n_similar_layers=1
reduct_type=avg
enhance_type=mmbeg
#enhance_type=mmbb
pos_enc_type=num
#pos_enc_type=emb

tokenizer_cfg_fpath=$config_dir_path/$tokenizer_cfg_fname
ranker_model_cfg_fpath=$config_dir_path/$ranker_model_cfg_fname


device=cpu
epochs=5
train_epoch_steps=20
val_epoch_steps=20
docs_batch_size=3
max_chunks_per_doc=3
pretrained_model_path=$train_encdec_root_path/encdechg-20241209_215611-inp128-lrs7x1-rdc_avg-enh_mmbb-step2-d256-h8


#device=cuda
#epochs=300
#train_epoch_steps=500
#val_epoch_steps=50
#docs_batch_size=20
#max_chunks_per_doc=5
#pretrained_model_path=$train_encdec_root_path/encdechg-20241209_215611-inp128-lrs7x1-rdc_avg-enh_mmbb-step2-d256-h8
##train_subdir="last"

learning_rate=0.0001

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_03_04_train_ranker_qrels.py \
  --ds-dir-paths $msmarco_data_path $fever_data_path \
  --train-root-path $train_ranker_root_path \
  --train-subdir "$train_subdir" \
  --tokenizer-cfg-fpath $tokenizer_cfg_fpath \
  --model-cfg-fpath $ranker_model_cfg_fpath \
  --model-level $model_level \
  --n-enc-layers $n_enc_layers \
  --n-dec-layers $n_dec_layers \
  --docs-batch-size $docs_batch_size \
  --max-chunks-per-doc $max_chunks_per_doc \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --pretrained-model-path $pretrained_model_path
#"

