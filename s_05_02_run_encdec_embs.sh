
code_path=$HOME/prog
data_path=$HOME/data
model_level=1
ds_dir_path=$data_path/ranker_embs_msmarco_fever
train_encdec_root_path=$data_path/train_mllm_encdec_${model_level}
#train_subdir=encdec-l1-20240918_063547-msmarco-fever
#out_ds_path=$data_path/encdec_embs_${model_level}_msmarco_fever
train_subdir=encdec-l1-20241005_175446-msmarco-fever
out_ds_path=$data_path/encdec_embs_${model_level}_msmarco_fever_v2
chunk_size=100

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
encdec_model_cfg_fpath=$config_dir_path/encdec_model_cfg_02.yaml

#device=cpu
#chunks_batch_size=3
#n_batches=10

device=cuda
chunks_batch_size=100
n_batches=0

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_05_02_run_encdec_embs.py \
  --ds-dir-path $ds_dir_path \
  --train-root-path $train_encdec_root_path \
  --train-subdir $train_subdir \
  --model-cfg-fpath $encdec_model_cfg_fpath \
  --model-level $model_level \
  --chunk-size $chunk_size \
  --batch-size $chunks_batch_size \
  --n-batches $n_batches \
  --device $device \
  --out-ds-path $out_ds_path
#"
