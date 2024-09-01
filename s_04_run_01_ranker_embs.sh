
code_path=$HOME/prog
data_path=$HOME/data
msmarco_data_path=$data_path/msmarco
fever_data_path=$data_path/fever
train_ranker_root_path=$data_path/train_mllm_ranker_qrels

emb_chunk_size=100

device=cpu
docs_batch_size=3
train_subdir=ranker-20240831_214832-msmarco-fever

#device=cuda
#docs_batch_size=100
#train_subdir=ranker-20240831_214832-msmarco-fever

mllm_src_path=$code_path/mllm
export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_04_run_01_ranker_embs.py \
  --ds-dir-paths $msmarco_data_path $fever_data_path \
  --train-root-path $train_ranker_root_path \
  --train-subdir $train_subdir \
  --emb-chunk-size $emb_chunk_size \
  --docs-batch-size $docs_batch_size \
  --device $device

