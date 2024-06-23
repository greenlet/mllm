
code_path=$HOME/prog
data_path=$HOME/data
ds_path=$data_path/wikipedia/20200501.en
out_path=$data_path/wiki_20200501_en/tok


mllm_src_path=$code_path/mllm
export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
python s_02_preproc_chunks.py \
  --ds-path $ds_path \
  --out-path $out_path

