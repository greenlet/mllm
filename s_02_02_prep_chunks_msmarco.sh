
code_path=$HOME/prog
data_path=$HOME/data
ds_path=$data_path/msmarco
out_path=$data_path/msmarco_chunks
emb_chunk_size=100
chunk_fixed_size_arg=""
chunk_fixed_size_arg="--chunk-fixed-size"
# max_docs=1000
max_docs=0


mllm_src_path=$code_path/mllm
export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1

#echo "
python s_02_02_prep_chunks_msmarco.py \
  --ds-path $ds_path \
  --emb-chunk-size $emb_chunk_size \
  $chunk_fixed_size_arg \
  --max-docs $max_docs \
  --out-path $out_path
#"

