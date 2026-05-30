# s_03_12_eval_mixed_decoder

## 1. Script And Run Info

- Script: `s_03_12_eval_mixed_decoder.py`
- Source script copied from: `eval_qna_per_dataset.py`
- Run command (system python, no conda):

```bash
cd /scratch/azureml/cr/j/9e6c96bb7ad54e1eaecc52cf51abb7e5/exe/wd
PYTHONPATH=. /usr/bin/python3 s_03_12_eval_mixed_decoder.py
```

- Completed log file: `/tmp/qna_eval_full.log`
- Completion status: `Done.` (exit code `0`)

## 2. Input Variables, Paths, And Constants

### 2.1 Paths

- `REPO_ROOT`: `/scratch/azureml/cr/j/9e6c96bb7ad54e1eaecc52cf51abb7e5/exe/wd`
- `DATA_PATH`: `/scratch/azureml/cr/j/9e6c96bb7ad54e1eaecc52cf51abb7e5/exe/wd/data`
- `TRAIN_ROOT`: `/scratch/azureml/cr/j/9e6c96bb7ad54e1eaecc52cf51abb7e5/exe/wd/data/train_mllm_encdec_bert`
- `RUN_DIR`:
  `/scratch/azureml/cr/j/9e6c96bb7ad54e1eaecc52cf51abb7e5/exe/wd/data/train_mllm_encdec_bert/mixeddecoder-20260530_113509-pre_mixeddecoder20260523180218-bertbaseuncased-d768-embEncCls-inp128-decQwen2.51.5b-msl400-dtypeBf16-sepF-pallF-eer4-ewn2x6-frzencF-dsQnaans-trn_lr5e-05_bs20_attdp0.1`
- `BEST_CKPT`:
  `/scratch/azureml/cr/j/9e6c96bb7ad54e1eaecc52cf51abb7e5/exe/wd/data/train_mllm_encdec_bert/mixeddecoder-20260530_113509-pre_mixeddecoder20260523180218-bertbaseuncased-d768-embEncCls-inp128-decQwen2.51.5b-msl400-dtypeBf16-sepF-pallF-eer4-ewn2x6-frzencF-dsQnaans-trn_lr5e-05_bs20_attdp0.1/best.pth`
- `MODEL_CFG_YAML`:
  `/scratch/azureml/cr/j/9e6c96bb7ad54e1eaecc52cf51abb7e5/exe/wd/data/train_mllm_encdec_bert/mixeddecoder-20260530_113509-pre_mixeddecoder20260523180218-bertbaseuncased-d768-embEncCls-inp128-decQwen2.51.5b-msl400-dtypeBf16-sepF-pallF-eer4-ewn2x6-frzencF-dsQnaans-trn_lr5e-05_bs20_attdp0.1/mixed_decoder_model_cfg.yaml`
- `CACHE_DIR`: `/scratch/azureml/cr/j/9e6c96bb7ad54e1eaecc52cf51abb7e5/exe/wd/data`
- Cache subdir used: `/scratch/azureml/cr/j/9e6c96bb7ad54e1eaecc52cf51abb7e5/exe/wd/data/qna_noanswer_cache`

### 2.2 Eval Hyperparameters

- `BATCH_SIZE = 10`
- `N_EVAL_BATCHES = 80`
- `RANDOM_SEED = 42`
- `DEVICE = cuda:0`
- `EMB_WIN_MAX = 6`
- `EMB_WIN_MIN = 2`
- Dataset tokenizer/chunk params used by loader in this eval:
  - `INP_LEN = 128`
  - `MAX_CHUNKS = 6`
  - `exclude_noanswer = True`
  - `sources = QNA_DATASETS_DEFAULT`

### 2.3 Model/Checkpoint Runtime Info

- Checkpoint keys:
  - `model`
  - `optimizer`
  - `scheduler`
  - `last_epoch`
  - `val_loss_min`
  - `optimizer_name`
  - `optimizer_params`
- `last_epoch = 25`
- `val_loss_min = 1.86125`
- Loaded parameter tensors: `Load 538`
- Model dtype/device for eval: `torch.bfloat16` on `cuda:0`

## 3. Dataset Analysis Report

### 3.1 Raw + Cached Loading Notes

- SQuAD v2 loaded raw: train `130319`, val `11873`
  - cached filtered: train `86821`, val `5928`
- NaturalQuestions loaded raw: train `307373`, val `7830`
  - cached filtered: train `152148`, val `5499`
- TriviaQA loaded: train `138384`, val `17944`
- NewsQA loaded raw: train `74160`, val `4212`
  - cached filtered: train `74160`, val `4212`
- MRQA loaded: train `190312`, val `22881`
- AdversarialQA loaded: train `30000`, val `3000`
- QuAC loaded raw dialogues: train `11567`, val `1000`
  - cached filtered: train `9430`, val `787`
- CoQA loaded raw dialogues: train `7199`, val `500`
  - cached filtered: train `7092`, val `497`

### 3.2 Dataset Size/Answer Stats Table

| Dataset | TrnN | ValN | AnsLen(trn) mean±std [min,max] | AnsLen(val) mean±std [min,max] | AggAnsRate (trn/val) |
|---|---:|---:|---|---|---|
| squad_v2 | 86821 | 5928 | 6.1±4.4 [2,31] | 5.7±3.7 [2,31] | 0.68/0.46 |
| natural_questions | 152148 | 5499 | 61.8±42.7 [2,100] | 57.2±43.3 [2,100] | 0.51/0.69 |
| triviaqa | 138384 | 17944 | 5.2±2.1 [2,16] | 5.2±2.1 [2,16] | 1.00/1.00 |
| newsqa | 74160 | 4212 | 7.3±5.9 [2,48] | 6.6±5.0 [2,35] | 1.00/1.00 |
| mrqa | 190312 | 22881 | 4.5±2.9 [2,48] | 4.2±1.8 [2,13] | 1.00/1.00 |
| adversarialqa | 30000 | 3000 | 6.5±7.7 [2,100] | 5.3±4.5 [2,48] | 1.00/1.00 |
| quac | 9430 | 787 | 19.2±12.0 [2,54] | 19.2±12.0 [3,49] | 0.81/0.77 |
| coqa | 7092 | 497 | 4.9±3.7 [2,28] | 4.5±2.8 [2,20] | 0.99/0.99 |

- Total filtered size: train `688347`, val `60748`

## 4. Evaluation Report (Loss/PPL)

### 4.1 Summary Table

| Dataset | Split | N | Loss mean | Loss std | PPL |
|---|---|---:|---:|---:|---:|
| squad_v2 | train | 86821 | 0.9453 | 0.3680 | 2.57 |
| squad_v2 | val | 5928 | 1.7573 | 0.5400 | 5.80 |
| natural_questions | train | 152148 | 1.0608 | 0.4367 | 2.89 |
| natural_questions | val | 5499 | 1.8583 | 0.5832 | 6.41 |
| triviaqa | train | 138384 | 1.9364 | 0.4807 | 6.93 |
| triviaqa | val | 17944 | 2.5948 | 0.5934 | 13.39 |
| newsqa | train | 74160 | 1.2466 | 0.4629 | 3.48 |
| newsqa | val | 4212 | 2.6708 | 0.5465 | 14.45 |
| mrqa | train | 190312 | 0.5731 | 0.2723 | 1.77 |
| mrqa | val | 22881 | 1.2422 | 0.4685 | 3.46 |
| adversarialqa | train | 30000 | 1.2722 | 0.4284 | 3.57 |
| adversarialqa | val | 3000 | 3.1200 | 0.6351 | 22.65 |
| quac | train | 9430 | 2.2373 | 0.4412 | 9.37 |
| quac | val | 787 | 2.5623 | 0.3892 | 12.97 |
| coqa | train | 7092 | 2.5684 | 0.5728 | 13.05 |
| coqa | val | 497 | 2.6983 | 0.5712 | 14.85 |

### 4.2 Overall And Gap Report

- OVERALL mean train loss: `1.4800`
- OVERALL mean val loss: `2.3130`
- OVERALL gap (val - train): `+0.8330`

Per-dataset train/val gap:

- squad_v2: `+0.8120` (val worse)
- natural_questions: `+0.7975` (val worse)
- triviaqa: `+0.6584` (val worse)
- newsqa: `+1.4241` (val worse)
- mrqa: `+0.6691` (val worse)
- adversarialqa: `+1.8478` (val worse)
- quac: `+0.3250` (val worse)
- coqa: `+0.1299` (val worse)

## 5. Important Observation Captured During Eval

The report includes the aggregator note used in the run:

- `QnaDatasetAgg maps position indices [0..N] directly to HF row indices, meaning the noanswer-filtered sub_ds.inds is bypassed in the aggregator path.`

This explains why `AggAnsRate` can be below `1.0` for some datasets/splits even with `exclude_noanswer=True` and cache files present.
