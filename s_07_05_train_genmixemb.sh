
code_path=$HOME/prog
data_path=$HOME/data

mllm_src_path=$code_path/mllm
config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fname=genmixemb_cfg_01_base.yaml

config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fpath=$config_dir_path/$model_cfg_fname
train_root_path=$data_path/train_mllm_genmixemb
train_root_path=$data_path/train_mllm_genmixemb_qna
train_encdec_root_path=$data_path/train_mllm_encdec_bert

#train_ds_type=wki
train_ds_type=qna
#train_ds_type=sum

model_name=bert-base-uncased
#model_name=bert-large-uncased
#toks_agg_type=brt
#toks_agg_type=pyr
toks_agg_type=cnv
#bert_agg_type=sep
#bert_agg_type=topcos
bert_agg_type=topdot
#bert_agg_n_subseq_toks=0
bert_agg_n_subseq_toks=2
bert_agg_n_subseq_toks=8
#pyr_agg_type=decim
#pyr_agg_type=matmul
#pyr_agg_type=avg
#pyr_agg_type=sub
#pyr_agg_type=topcos
#pyr_agg_type=topdot
pyr_agg_type=mxpl
pyr_agg_step=2
#pyr_agg_step=4
#pyr_agg_n_levels=1
pyr_agg_n_levels=2
#pyr_agg_n_levels=3
#pyr_agg_n_layers_per_level=2
#pyr_agg_n_layers_per_level=4
pyr_agg_n_layers_per_level=2
#pyr_share_layer_weights=false
pyr_share_layer_weights=true
#train_agg_model=false
train_agg_model=true
#share_agg_enc_token_embeds=false
share_agg_enc_token_embeds=true
add_token_type_ids=false
#add_token_type_ids=true
join_ctx_que_agg=false
#ctx_que_prompt_type=tok
ctx_que_prompt_type=cq
#ctx_que_prompt_type=qc
#ctx_que_prompt_type=cqqc
dec_expert_type=non
moe_experts_num=0

mask_tokens=false
#mask_tokens=true
mask_sep_freq=0.5
mask_sep_frac=0.15
mask_seq_freq=0.5
mask_seq_max_frac=0.2
mask_seq_max_len=20


n_toks_min=20
max_inp_toks=100
max_inp_toks=256
max_inp_toks=384
#max_inp_toks=512
max_out_toks=50


train_ds_type=qna
#model_name=bert-base-uncased
#toks_agg_type=cnv
##cnv_n_levels=1
#cnv_n_levels=2
#cnv_n_layers_per_level=1
#cnv_conv_kernel_size=3
#cnv_pool_kernel_size=2
#cnv_pool_stride=2
##cnv_share_layer_weights=false
#cnv_share_layer_weights=true
#train_agg_model=true
#add_token_type_ids=false
#share_agg_enc_token_embeds=true
#join_ctx_que_agg=false
#ctx_que_prompt_type=cq


train_ds_type=wki
model_name=bert-base-large
toks_agg_type=brt
bert_agg_n_subseq_toks=0
train_agg_model=true
add_token_type_ids=false
share_agg_enc_token_embeds=true
join_ctx_que_agg=false
ctx_que_prompt_type=cq
mask_tokens=false
self_supervise_type=nxttok
n_toks_min=50
max_inp_toks=448
max_out_toks=36


train_root_path=$data_path/train_mllm_genmixemb_wki
train_ds_type=wki
model_name=gpt2
#model_name=gpt2-large
train_agg_model=true
toks_agg_type=cnv
ctx_que_prompt_type=cq
cnv_n_levels=3
cnv_n_levels=6
cnv_n_layers_per_level=1
cnv_conv_kernel_size=3
cnv_pool_kernel_size=2
cnv_pool_stride=2
#cnv_share_layer_weights=false
cnv_share_layer_weights=true
n_toks_min=20
max_inp_toks=1024
# max_inp_toks=448
#max_inp_toks=128
#max_inp_toks=64
#max_out_toks=256
max_out_toks=128
#max_out_toks=16

#train_root_path=$data_path/train_mllm_genmixemb_wki
#train_ds_type=wki
#model_name=gpt2
#train_agg_model=true
#toks_agg_type=brt
##bert_agg_type=sep
##bert_agg_type=topcos
#bert_agg_type=topdot
##bert_agg_n_subseq_toks=2
#bert_agg_n_subseq_toks=8
#ctx_que_prompt_type=cq
#n_toks_min=20
##max_inp_toks=1024
#max_inp_toks=512
#max_out_toks=128


train_root_path=$data_path/train_mllm_genmixemb_qna
train_ds_type=qna
model_name=gpt2
#model_name=gpt2-large
train_agg_model=true
toks_agg_type=cnv
ctx_que_prompt_type=cq
cnv_n_levels=3
#cnv_n_levels=6
cnv_n_layers_per_level=1
cnv_conv_kernel_size=3
cnv_pool_kernel_size=2
cnv_pool_stride=2
cnv_share_layer_weights=false
#cnv_share_layer_weights=true
n_toks_min=20
max_inp_toks=768
# max_inp_toks=448
#max_inp_toks=128
#max_inp_toks=64
#max_out_toks=256
max_out_toks=128
#max_out_toks=16

device=cpu
epochs=5
train_epoch_steps=20
val_epoch_steps=20
batch_size=5

#pretrained_model_path=$train_encdec_root_path/encdecbert-20250131_223521-bert-base-uncased-d768-emb_cls-inp128-lrs7x1-enh_mmbb-step2-h12-dp0-t0.0
#pretrained_model_path=$train_root_path/genmixemb-20250713_202718-bertbaseuncased-d768-mxo50-aggBrt-sub0-dsWki-tmax100-tragF
#pretrained_model_path=$train_root_path/genmixemb-20250721_083250-bertbaseuncased-d768-mxo50-aggPyr-agtDecim-stp0-lvl1-lrs2-dsWki-tmax256-tragF-nxtsnt
#pretrained_model_path=$train_root_path/genmixemb-20250721_212402-bertbaseuncased-d768-mxo50-aggPyr-agtDecim-stp2-lvl2-lrs2-dsWki-tmax512-tragT-nxtsnt

#pretrained_model_path=$train_root_path/genmixemb-20250726_122548-bertbaseuncased-d768-mxi384-mxo50-dsQna-ttidF
#pretrained_model_path=$train_root_path/genmixemb-20250810_125920-pre_genmixemb20250726122548-bertbaseuncased-d768-mxi384-mxo50-aggBrt-sub2-agtTopdot-dsQna-tragT-shemT-ttidF-jcqF
#pretrained_model_path=$train_root_path/genmixemb-20250815_220237-pre_genmixemb20250726122548-bertbaseuncased-d768-mxi384-mxo50-aggPyr-agtMxpl-stp2-lvl1-lrs2-dsQna-tragT-shemT-ttidF-cqprCq
#pretrained_model_path=$train_root_path/genmixemb-20250817_201509-pre_genmixemb20250726122548-bertbaseuncased-d768-mxi384-mxo50-aggCnv-lvl1-lrs1-cksz3-pksz2-pst2-dsQna-tragT-ttidF-cqprCq

#pretrained_model_path=$pretrained_model_path/best.pth


device=cuda
epochs=700
train_epoch_steps=500
val_epoch_steps=50
#batch_size=30
#batch_size=25
#batch_size=20
#batch_size=15
batch_size=10
#batch_size=5
#batch_size=1
#train_subdir=last

#learning_rate=0.0001
learning_rate=0.00005
#learning_rate=0.00001
#learning_rate=0.000005
random_seed=200

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_07_05_train_genmixemb.py \
  --data-path $data_path \
  --train-root-path $train_root_path \
  --pretrained-model-path "$pretrained_model_path" \
  --train-subdir "$train_subdir" \
  --train-ds-type "$train_ds_type" \
  --model-cfg-fpath $model_cfg_fpath \
  --model-name $model_name \
  --max-inp-toks $max_inp_toks \
  --max-out-toks $max_out_toks \
  --toks-agg-type $toks_agg_type \
  --bert-agg-type $bert_agg_type \
  --bert-agg-n-subseq-toks $bert_agg_n_subseq_toks \
  --pyr-agg-type $pyr_agg_type \
  --pyr-agg-step $pyr_agg_step \
  --pyr-agg-n-levels $pyr_agg_n_levels \
  --pyr-agg-n-layers-per-level $pyr_agg_n_layers_per_level \
  --pyr-share-layer-weights $pyr_share_layer_weights \
  --cnv-n-levels $cnv_n_levels \
  --cnv-n-layers-per-level $cnv_n_layers_per_level \
  --cnv-conv-kernel-size $cnv_conv_kernel_size \
  --cnv-pool-kernel-size $cnv_pool_kernel_size \
  --cnv-pool-stride $cnv_pool_stride \
  --cnv-share-layer-weights $cnv_share_layer_weights \
  --train-agg-model $train_agg_model \
  --self-supervise-type $self_supervise_type \
  --share-agg-enc-token-embeds $share_agg_enc_token_embeds \
  --add-token-type-ids $add_token_type_ids \
  --join-ctx-que-agg $join_ctx_que_agg \
  --ctx-que-prompt-type $ctx_que_prompt_type \
  --dec-expert-type $dec_expert_type \
  --moe-experts-num $moe_experts_num \
  --n-toks-min $n_toks_min \
  --mask-tokens $mask_tokens \
  --mask-sep-freq $mask_sep_freq \
  --mask-sep-frac $mask_sep_frac \
  --mask-seq-freq $mask_seq_freq \
  --mask-seq-max-frac $mask_seq_max_frac \
  --mask-seq-max-len $mask_seq_max_len \
  --batch-size $batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --random-seed $random_seed
#"

