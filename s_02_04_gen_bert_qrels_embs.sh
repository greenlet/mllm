
code_path=$HOME/prog
data_path=$HOME/data
msmarco_data_path=$data_path/msmarco
fever_data_path=$data_path/fever
out_ds_path=$data_path/ranker_embs_msmarco_fever_bert
tokens_chunk_size=512

mllm_src_path=$code_path/mllm

device=cpu
batch_size=5
n_docs=100
n_qs=100

#device=cuda
#batch_size=5
#n_docs=0
#n_qs=0

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_02_04_gen_bert_qrels_embs.py \
  --ds-dir-paths $msmarco_data_path $fever_data_path \
  --out-ds-path $out_ds_path \
  --tokens-chunk-size $tokens_chunk_size \
  --batch-size $batch_size \
  --n-docs $n_docs \
  --n-qs $n_qs \
  --device $device
#"
