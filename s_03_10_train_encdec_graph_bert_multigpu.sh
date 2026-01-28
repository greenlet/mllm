
# code_path=$HOME/prog
# data_path=$HOME/data
# mllm_src_path=$code_path/mllm
code_path=$AZUREML_CR_EXECUTION_WORKING_DIR_PATH
data_path=$code_path/data
mllm_src_path=$code_path

#wiki_ds_name=20200501.en
wiki_ds_name=20220301.en

config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fname=encdec_graph_bert_cfg_01.yaml

config_dir_path=$mllm_src_path/mllm/config/cfg
model_cfg_fpath=$config_dir_path/$model_cfg_fname
train_root_path=$data_path/train_mllm_encdec_bert

bert_model_name=bert-base-uncased
bert_emb_type=cls
inp_len=128
dec_n_layers=0
dec_n_similar_layers=1
dec_enhance_type=mmbb
dec_dropout_rate=0
mask_tokens=false
mask_sep_freq=0.5
mask_sep_frac=0.15
mask_seq_freq=0.5
mask_seq_max_frac=0.2
mask_seq_max_len=20
mask_n_last_toks=0
share_enc_dec_proj_weights=false

emb_middle_type=graph
# n_graph_layers=2
# gnn_hidden_dim=-1
# gnn_conv_name='GCNConv'
# gnn_conv_params='{"normalize": true, "bias": false}'
n_graph_layers=8
# gnn_hidden_dim=1536
gnn_hidden_dim=-1
gnn_conv_name='ChebConv'
gnn_conv_params='{"K": 3, "bias": true}'

emb_middle_type=attn
n_emb_attn_layers=8

emb_middle_type=mlp
emb_mlp_window_size=3
emb_mlp_n_window_layers=1
emb_mlp_n_out_layers=1
emb_mlp_act_fn='gelu'

emb_middle_type=rnn
emb_rnn_n_layers=2
emb_rnn_hidden_dim=-1
emb_rnn_input_order='cp'
emb_rnn_cell_name='LSTM'
emb_rnn_cell_params='{"bidirectional": false, "dropout": 0.0}'

cite_toks_target_weight=1
cite_toks_target_type='all'
cite_toks_target_scale=1
cite_embs_target_weight=1
# cite_embs_target_type='cos'
# cite_embs_target_scale=20.0
# cite_embs_target_type='mse'
# cite_embs_target_scale=100.0
cite_embs_target_type='sqrt'
cite_embs_target_scale=10.0
# cite_embs_target_type='r2'
# cite_embs_target_scale=1.0
input_toks_target_weight=1
input_toks_target_scale=1


#pretrained_model_path=$train_root_path/encdecbert-20250131_223521-bert-base-uncased-d768-emb_cls-inp128-lrs7x1-enh_mmbb-step2-h12-dp0-t0.0
pretrained_model_path=$train_root_path/encdecbert-20260110_193915-bertbaseuncased-d768-embCls-inp128-lrs7x1-enhMmbb-step2-h12-dp0-t0.0




# device=cpu
# epochs=5
# train_epoch_steps=20
# val_epoch_steps=20
# docs_batch_size=5
# world_size=1

device=cuda
epochs=700
train_epoch_steps=500
val_epoch_steps=50
docs_batch_size=40
world_size=4


# learning_rate=0.0001
learning_rate=0.00005
#learning_rate=0.00001
random_seed=200

optimizer_name='AdamW'
optimizer_params='{}'
# learning_rate_scheduler_name='ReduceLROnPlateau'
# learning_rate_scheduler_params='{"mode": "min", "factor": 0.5, "patience": 5, "threshold": 1e-6, "min_lr": 1e-8}'
learning_rate_scheduler_name='CosineAnnealingLR'
learning_rate_scheduler_params='{"T_max": 100, "eta_min": 1e-8}'

export PYTHONPATH=$PYTHONPATH:$mllm_src_path

cd "$mllm_src_path" || exit 1
#echo "
python s_03_10_train_encdec_graph_bert_multigpu.py \
  --data-path $data_path \
  --wiki-ds-name $wiki_ds_name \
  --train-root-path $train_root_path \
  --train-subdir "$train_subdir" \
  --model-cfg-fpath $model_cfg_fpath \
  --bert-model-name $bert_model_name \
  --bert-emb-type $bert_emb_type \
  --inp-len $inp_len \
  --dec-enhance-type $dec_enhance_type \
  --dec-n-layers $dec_n_layers \
  --dec-n-similar-layers $dec_n_similar_layers \
  --dec-dropout-rate $dec_dropout_rate \
  --share-enc-dec-proj-weights $share_enc_dec_proj_weights \
  --emb-middle-type $emb_middle_type \
  --n-graph-layers $n_graph_layers \
  --gnn-hidden-dim $gnn_hidden_dim \
  --gnn-conv-name $gnn_conv_name \
  --gnn-conv-params "$gnn_conv_params" \
  --n-emb-attn-layers $n_emb_attn_layers \
  --emb-mlp-window-size $emb_mlp_window_size \
  --emb-mlp-n-window-layers $emb_mlp_n_window_layers \
  --emb-mlp-n-out-layers $emb_mlp_n_out_layers \
  --emb-mlp-act-fn $emb_mlp_act_fn \
  --emb-rnn-n-layers $emb_rnn_n_layers \
  --emb-rnn-hidden-dim $emb_rnn_hidden_dim \
  --emb-rnn-input-order $emb_rnn_input_order \
  --emb-rnn-cell-name $emb_rnn_cell_name \
  --emb-rnn-cell-params "$emb_rnn_cell_params" \
  --mask-tokens $mask_tokens \
  --mask-sep-freq $mask_sep_freq \
  --mask-sep-frac $mask_sep_frac \
  --mask-seq-freq $mask_seq_freq \
  --mask-seq-max-frac $mask_seq_max_frac \
  --mask-seq-max-len $mask_seq_max_len \
  --mask-n-last-toks $mask_n_last_toks \
  --cite-toks-target-weight $cite_toks_target_weight \
  --cite-toks-target-type $cite_toks_target_type \
  --cite-toks-target-scale $cite_toks_target_scale \
  --cite-embs-target-weight $cite_embs_target_weight \
  --cite-embs-target-type $cite_embs_target_type \
  --cite-embs-target-multiplier $cite_embs_target_scale \
  --input-toks-target-weight $input_toks_target_weight \
  --input-toks-target-scale $input_toks_target_scale \
  --docs-batch-size $docs_batch_size \
  --device $device \
  --epochs $epochs \
  --learning-rate $learning_rate \
  --optimizer-name $optimizer_name \
  --optimizer-params "$optimizer_params" \
  --learning-rate-scheduler-name $learning_rate_scheduler_name \
  --learning-rate-scheduler-params "$learning_rate_scheduler_params" \
  --train-epoch-steps $train_epoch_steps \
  --val-epoch-steps $val_epoch_steps \
  --random-seed $random_seed \
  --pretrained-model-path "$pretrained_model_path" \
  --world-size $world_size
#"

