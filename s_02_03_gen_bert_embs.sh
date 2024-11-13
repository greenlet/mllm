
code_path=$HOME/prog
data_path=$HOME/data
mllm_src_path=$code_path/mllm
msmarco_data_path=$data_path/msmarco
fever_data_path=$data_path/fever
tokens_chunk_size=512

mllm_src_path=$code_path/mllm

device=cpu
batch_size=5
max_docs=100

#device=cuda
#batch_size=5
#max_docs=100

#device=cuda
#batch_size=7
#max_docs=1000

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_02_04_gen_bert_qrels_embs.py \
  --data-path $data_path \
  --wiki-ds-name $wiki_ds_name \
  --out-ds-path $out_wiki_ds_path \
  --tokens-chunk-size $tokens_chunk_size \
  --max-chunks-per-doc $max_chunks_per_doc \
  --batch-size $batch_size \
  --device $device \
  --max-docs $max_docs
#"
