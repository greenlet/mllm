
code_path=$HOME/prog
data_path=$HOME/data
msmarco_data_path=$data_path/msmarco
fever_data_path=$data_path/fever
train_ranker_root_path=$data_path/train_mllm_ranker_qrels_0
train_encdec_root_path=$data_path/train_mllm_encdec_0

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
tokenizer_cfg_fname=tokenizer_cfg_01.yaml
ranker_model_cfg_fname=ranker_model_cfg_01.yaml
model_level=0
n_enc_layers=3
n_dec_layers=3

tokenizer_cfg_fpath=$config_dir_path/$tokenizer_cfg_fname
ranker_model_cfg_fpath=$config_dir_path/$ranker_model_cfg_fname


#device=cpu
#epochs=5
#train_epoch_steps=20
#val_epoch_steps=20
#docs_batch_size=3
#max_chunks_per_doc=3
#pretrained_model_path=$train_encdec_root_path/encdec-20241018_092135-wiki_20200501_en-ch_100_fixed


device=cuda
epochs=300
train_epoch_steps=500
val_epoch_steps=50
docs_batch_size=20
max_chunks_per_doc=5
pretrained_model_path=$train_encdec_root_path/encdec-lvl0-20241029_140645-wiki_20200501_en-ch_100_fixed-enc-lrs3-embmatFalse-d256-h8-dec-lrs3-seqlen100-d256-h8-vocdecTrue
#train_subdir="last"

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

