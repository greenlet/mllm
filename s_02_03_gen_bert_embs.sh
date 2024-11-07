
code_path=$HOME/prog
data_path=$HOME/data
wiki_ds_name=20200501.en
out_wiki_ds_path=$data_path/wiki_20220301_en_bert
tokens_chunk_size=512
max_chunks_per_doc=10

mllm_src_path=$code_path/mllm

device=cpu
batch_size=5
max_docs=1000

#device=cuda
#batch_size=50
max_docs=0

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_02_03_gen_bert_embs.py \
  --data-path $data_path \
  --wiki-ds-name $wiki_ds_name \
  --out-ds-path $out_wiki_ds_path \
  --tokens-chunk-size $tokens_chunk_size \
  --max-chunks-per-doc $max_chunks_per_doc \
  --batch-size $batch_size \
  --device $device \
  --max_docs $max_docs
#"
