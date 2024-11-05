
code_path=$HOME/prog
data_path=$HOME/data
wiki_ds_path=$data_path/wikipedia/20200501.en
out_path=$data_path/wiki_20200501_en_bert
emb_chunk_size=512

mllm_src_path=$code_path/mllm

device=cpu
batch_size=5

#device=cuda
#batch_size=50

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_02_03_gen_bert_embs.py \
  --wiki-ds-path $wiki_ds_path \
  --out-path $out_path \
  --emb-chunk-size $emb_chunk_size \
  --device $device \
  --batch-size $batch_size
#"
