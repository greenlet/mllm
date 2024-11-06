
code_path=$HOME/prog
data_path=$HOME/data
wiki_ds_name=20220301.en
out_wiki_ds_path=$data_path/wiki_20220301_en_bert
max_tokens_chunk_size=512

mllm_src_path=$code_path/mllm

device=cpu
batch_size=5

#device=cuda
#batch_size=50

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_02_03_gen_bert_embs.py \
  --data-path $data_path \
  --wiki-ds-name $wiki_ds_name \
  --out-ds-path $out_wiki_ds_path \
  --max-tokens-chunk-size $max_tokens_chunk_size \
  --batch-size $batch_size \
  --device $device
#"
