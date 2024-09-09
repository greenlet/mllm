
code_path=$HOME/prog
data_path=$HOME/data
msmarco_data_path=$data_path/msmarco
fever_data_path=$data_path/fever
train_ranker_root_path=$data_path/train_mllm_ranker_qrels
out_ds_path=$data_path/ranker_embs_msmarco_fever

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg_v001
tokenizer_cfg_fname=tokenizer_cfg_01.yaml
model_cfg_fname=ranker_model_cfg_01.yaml

tokenizer_cfg_fpath=$config_dir_path/$tokenizer_cfg_fname
model_cfg_fpath=$config_dir_path/$model_cfg_fname

#device=cpu
#batch_size=10
#n_docs=10
#n_qs=30
#train_subdir=ranker-20240903_215749-msmarco-fever

device=cuda
batch_size=2500
n_docs=0
n_qs=0
train_subdir=ranker-20240903_215749-msmarco-fever

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_04_run_01_ranker_embs.py \
  --ds-dir-paths $msmarco_data_path $fever_data_path \
  --train-root-path $train_ranker_root_path \
  --train-subdir $train_subdir \
  --tokenizer-cfg-fpath $tokenizer_cfg_fpath \
  --model-cfg-fpath $model_cfg_fpath \
  --batch-size $batch_size \
  --n-docs $n_docs \
  --n-qs $n_qs \
  --device $device \
  --out-ds-path $out_ds_path
#"
