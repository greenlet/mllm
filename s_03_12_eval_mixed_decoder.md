# s_03_12_eval_mixed_decoder

## Run 1 (2026-05-31) — Baseline Before Fixed QnaAns Indexing

### Script And Run Info

- Script: `s_03_12_eval_mixed_decoder.py`
- Run command:

```bash
cd /scratch/azureml/cr/j/2ae1c2a5eece41e9a5c302f75176aaba/exe/wd
PYTHONPATH=. /usr/bin/python3 s_03_12_eval_mixed_decoder.py
```

- Log: `/tmp/qna_eval_full.log`
- Status: `Done.` (exit code `0`)

### Dataset Analysis Summary

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

### Evaluation Summary (Loss/PPL)

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

- OVERALL mean train loss: `1.4800`
- OVERALL mean val loss: `2.3130`
- OVERALL gap (val - train): `+0.8330`

## Run 2 (2026-06-01) — Fixed QnaAns Dataset Indexing (Filtered NoAnswer)

### Script And Run Info

- Script: `s_03_12_eval_mixed_decoder.py`
- Run command:

```bash
cd /scratch/azureml/cr/j/2ae1c2a5eece41e9a5c302f75176aaba/exe/wd
PYTHONPATH=. /usr/bin/python3 -u s_03_12_eval_mixed_decoder.py | tee /tmp/qna_eval_2026-06-01.log
```

- Log: `/tmp/qna_eval_2026-06-01.log`
- Status: `Done.` (exit code `0`)
- Checkpoint: `mixeddecoder-20260531_191822-pre_mixeddecoder20260523180218-...-dsQnaans-.../best.pth`

### Dataset Analysis Summary

| Dataset | TrnN | ValN | AnsLen(trn) mean±std [min,max] | AnsLen(val) mean±std [min,max] | AggAnsRate (trn/val) |
|---|---:|---:|---|---|---|
| squad_v2 | 86821 | 5928 | 6.1±4.4 [2,31] | 5.8±4.0 [2,31] | 0.68/0.46 |
| natural_questions | 152148 | 5499 | 60.3±43.2 [2,100] | 55.5±43.3 [2,100] | 0.51/0.69 |
| triviaqa | 138384 | 17944 | 5.5±3.7 [2,73] | 5.3±2.2 [2,21] | 1.00/1.00 |
| newsqa | 74160 | 4212 | 7.3±5.9 [2,48] | 6.6±5.0 [2,35] | 1.00/1.00 |
| mrqa | 190312 | 22881 | 4.5±2.9 [2,48] | 4.2±1.8 [2,13] | 1.00/1.00 |
| adversarialqa | 30000 | 3000 | 6.5±7.7 [2,100] | 5.3±4.5 [2,48] | 1.00/1.00 |
| quac | 9430 | 787 | 20.1±11.6 [2,54] | 19.4±11.8 [2,59] | 0.83/0.78 |
| coqa | 7092 | 497 | 4.4±3.0 [2,21] | 4.5±2.7 [2,19] | 0.99/0.99 |

- Total filtered size: train `688347`, val `60748`

### Evaluation Summary (Loss/PPL)

| Dataset | Split | N | Loss mean | Loss std | PPL |
|---|---|---:|---:|---:|---:|
| squad_v2 | train | 86821 | 1.0719 | 0.3926 | 2.92 |
| squad_v2 | val | 5928 | 2.4990 | 0.5450 | 12.17 |
| natural_questions | train | 152148 | 0.9171 | 0.3552 | 2.50 |
| natural_questions | val | 5499 | 1.5614 | 0.4679 | 4.77 |
| triviaqa | train | 138384 | 1.8933 | 0.4362 | 6.64 |
| triviaqa | val | 17944 | 2.6066 | 0.5506 | 13.55 |
| newsqa | train | 74160 | 1.2643 | 0.4696 | 3.54 |
| newsqa | val | 4212 | 2.5500 | 0.5708 | 12.81 |
| mrqa | train | 190312 | 0.5334 | 0.3064 | 1.70 |
| mrqa | val | 22881 | 1.1976 | 0.4528 | 3.31 |
| adversarialqa | train | 30000 | 1.2434 | 0.4229 | 3.47 |
| adversarialqa | val | 3000 | 2.9392 | 0.5621 | 18.90 |
| quac | train | 9430 | 2.0978 | 0.3926 | 8.15 |
| quac | val | 787 | 2.4692 | 0.4110 | 11.81 |
| coqa | train | 7092 | 2.4571 | 0.6575 | 11.67 |
| coqa | val | 497 | 2.5015 | 0.5995 | 12.20 |

- OVERALL mean train loss: `1.4348`
- OVERALL mean val loss: `2.2906`
- OVERALL gap (val - train): `+0.8558`

## Run 3 (2026-06-03) — Latest QnaansCite-Trained Checkpoint, Evaluated On Qnaans

### Script And Run Info

- Script: `s_03_12_eval_mixed_decoder.py`
- Run command:

```bash
cd /scratch/azureml/cr/j/69afa6bb42064469926aa31f99dac9a6/exe/wd
PYTHONPATH=. /usr/bin/python3 -u s_03_12_eval_mixed_decoder.py | tee /tmp/qna_eval_2026-06-03_qnaanscite.log
```

- Python: `Python 3.10.12` (system python)
- Log: `/tmp/qna_eval_2026-06-03_qnaanscite.log`
- Status: `Done.` (exit code `0`)
- Checkpoint: `mixeddecoder-20260602_213658-pre_mixeddecoder20260523180218-...-dsQnaanscite-.../best.pth`

### Dataset Analysis Summary

| Dataset | TrnN | ValN | AnsLen(trn) mean±std [min,max] | AnsLen(val) mean±std [min,max] | AggAnsRate (trn/val) |
|---|---:|---:|---|---|---|
| squad_v2 | 86821 | 5928 | 6.1±4.4 [2,31] | 5.9±4.1 [2,31] | 0.68/0.46 |
| natural_questions | 152148 | 5499 | 61.3±43.1 [2,100] | 53.7±43.2 [2,100] | 0.51/0.69 |
| triviaqa | 138384 | 17944 | 5.5±3.8 [2,70] | 5.4±2.4 [2,24] | 1.00/1.00 |
| newsqa | 74160 | 4212 | 7.3±5.9 [2,48] | 6.6±5.0 [2,35] | 1.00/1.00 |
| mrqa | 190312 | 22881 | 4.5±2.9 [2,48] | 4.2±1.8 [2,13] | 1.00/1.00 |
| adversarialqa | 30000 | 3000 | 6.5±7.7 [2,100] | 5.3±4.5 [2,48] | 1.00/1.00 |
| quac | 9430 | 787 | 19.7±12.3 [2,51] | 19.5±11.6 [2,59] | 0.83/0.79 |
| coqa | 7092 | 497 | 4.7±3.7 [2,32] | 4.6±3.1 [2,26] | 0.98/1.00 |

- Total filtered size: train `688347`, val `60748`

### Evaluation Summary (Loss/PPL)

| Dataset | Split | N | Loss mean | Loss std | PPL |
|---|---|---:|---:|---:|---:|
| squad_v2 | train | 86821 | 1.8582 | 0.5440 | 6.41 |
| squad_v2 | val | 5928 | 2.4678 | 0.5178 | 11.80 |
| natural_questions | train | 152148 | 1.3438 | 0.4075 | 3.83 |
| natural_questions | val | 5499 | 1.8091 | 0.4902 | 6.11 |
| triviaqa | train | 138384 | 2.6088 | 0.4797 | 13.58 |
| triviaqa | val | 17944 | 2.9165 | 0.6354 | 18.48 |
| newsqa | train | 74160 | 1.9961 | 0.4833 | 7.36 |
| newsqa | val | 4212 | 2.6918 | 0.4538 | 14.76 |
| mrqa | train | 190312 | 1.1088 | 0.3981 | 3.03 |
| mrqa | val | 22881 | 1.3998 | 0.4499 | 4.05 |
| adversarialqa | train | 30000 | 1.9796 | 0.4487 | 7.24 |
| adversarialqa | val | 3000 | 2.9224 | 0.5230 | 18.59 |
| quac | train | 9430 | 2.3423 | 0.4174 | 10.40 |
| quac | val | 787 | 2.4565 | 0.3802 | 11.66 |
| coqa | train | 7092 | 2.7320 | 0.5058 | 15.36 |
| coqa | val | 497 | 2.6830 | 0.5616 | 14.63 |

- OVERALL mean train loss: `1.9962`
- OVERALL mean val loss: `2.4184`
- OVERALL gap (val - train): `+0.4222`

## Joined Comparison Table (2026-06-01 vs 2026-06-03)

### Per Dataset/Split: Loss Mean, Loss Std, PPL, And Deltas

Delta definition: `Run3 - Run2`.

| Dataset | Split | Loss mean (06-01) | Loss mean (06-03) | Δ Loss mean | Loss std (06-01) | Loss std (06-03) | Δ Loss std | PPL (06-01) | PPL (06-03) | Δ PPL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| squad_v2 | train | 1.0719 | 1.8582 | +0.7863 | 0.3926 | 0.5440 | +0.1514 | 2.92 | 6.41 | +3.49 |
| squad_v2 | val | 2.4990 | 2.4678 | -0.0312 | 0.5450 | 0.5178 | -0.0272 | 12.17 | 11.80 | -0.37 |
| natural_questions | train | 0.9171 | 1.3438 | +0.4267 | 0.3552 | 0.4075 | +0.0523 | 2.50 | 3.83 | +1.33 |
| natural_questions | val | 1.5614 | 1.8091 | +0.2477 | 0.4679 | 0.4902 | +0.0223 | 4.77 | 6.11 | +1.34 |
| triviaqa | train | 1.8933 | 2.6088 | +0.7155 | 0.4362 | 0.4797 | +0.0435 | 6.64 | 13.58 | +6.94 |
| triviaqa | val | 2.6066 | 2.9165 | +0.3099 | 0.5506 | 0.6354 | +0.0848 | 13.55 | 18.48 | +4.93 |
| newsqa | train | 1.2643 | 1.9961 | +0.7318 | 0.4696 | 0.4833 | +0.0137 | 3.54 | 7.36 | +3.82 |
| newsqa | val | 2.5500 | 2.6918 | +0.1418 | 0.5708 | 0.4538 | -0.1170 | 12.81 | 14.76 | +1.95 |
| mrqa | train | 0.5334 | 1.1088 | +0.5754 | 0.3064 | 0.3981 | +0.0917 | 1.70 | 3.03 | +1.33 |
| mrqa | val | 1.1976 | 1.3998 | +0.2022 | 0.4528 | 0.4499 | -0.0029 | 3.31 | 4.05 | +0.74 |
| adversarialqa | train | 1.2434 | 1.9796 | +0.7362 | 0.4229 | 0.4487 | +0.0258 | 3.47 | 7.24 | +3.77 |
| adversarialqa | val | 2.9392 | 2.9224 | -0.0168 | 0.5621 | 0.5230 | -0.0391 | 18.90 | 18.59 | -0.31 |
| quac | train | 2.0978 | 2.3423 | +0.2445 | 0.3926 | 0.4174 | +0.0248 | 8.15 | 10.40 | +2.25 |
| quac | val | 2.4692 | 2.4565 | -0.0127 | 0.4110 | 0.3802 | -0.0308 | 11.81 | 11.66 | -0.15 |
| coqa | train | 2.4571 | 2.7320 | +0.2749 | 0.6575 | 0.5058 | -0.1517 | 11.67 | 15.36 | +3.69 |
| coqa | val | 2.5015 | 2.6830 | +0.1815 | 0.5995 | 0.5616 | -0.0379 | 12.20 | 14.63 | +2.43 |

### Per Dataset: Train-Val Gap Comparison

Gap definition: `val loss - train loss`.

| Dataset | Gap (06-01) | Gap (06-03) | Δ Gap |
|---|---:|---:|---:|
| squad_v2 | +1.4270 | +0.6095 | -0.8175 |
| natural_questions | +0.6443 | +0.4653 | -0.1790 |
| triviaqa | +0.7133 | +0.3077 | -0.4056 |
| newsqa | +1.2857 | +0.6957 | -0.5900 |
| mrqa | +0.6642 | +0.2910 | -0.3732 |
| adversarialqa | +1.6958 | +0.9428 | -0.7530 |
| quac | +0.3714 | +0.1142 | -0.2572 |
| coqa | +0.0444 | -0.0490 | -0.0934 |

### Dataset Analysis Comparison: AggAnsRate

| Dataset | AggAnsRate trn/val (06-01) | AggAnsRate trn/val (06-03) |
|---|---|---|
| squad_v2 | 0.68/0.46 | 0.68/0.46 |
| natural_questions | 0.51/0.69 | 0.51/0.69 |
| triviaqa | 1.00/1.00 | 1.00/1.00 |
| newsqa | 1.00/1.00 | 1.00/1.00 |
| mrqa | 1.00/1.00 | 1.00/1.00 |
| adversarialqa | 1.00/1.00 | 1.00/1.00 |
| quac | 0.83/0.78 | 0.83/0.79 |
| coqa | 0.99/0.99 | 0.98/1.00 |

## Joined Comparison Table (2026-05-31 vs 2026-06-01)

### Per Dataset/Split: Loss Mean, Loss Std, PPL, And Deltas

Delta definition: `Run2 - Run1`.

| Dataset | Split | Loss mean (05-31) | Loss mean (06-01) | Δ Loss mean | Loss std (05-31) | Loss std (06-01) | Δ Loss std | PPL (05-31) | PPL (06-01) | Δ PPL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| squad_v2 | train | 0.9453 | 1.0719 | +0.1266 | 0.3680 | 0.3926 | +0.0246 | 2.57 | 2.92 | +0.35 |
| squad_v2 | val | 1.7573 | 2.4990 | +0.7417 | 0.5400 | 0.5450 | +0.0050 | 5.80 | 12.17 | +6.37 |
| natural_questions | train | 1.0608 | 0.9171 | -0.1437 | 0.4367 | 0.3552 | -0.0815 | 2.89 | 2.50 | -0.39 |
| natural_questions | val | 1.8583 | 1.5614 | -0.2969 | 0.5832 | 0.4679 | -0.1153 | 6.41 | 4.77 | -1.64 |
| triviaqa | train | 1.9364 | 1.8933 | -0.0431 | 0.4807 | 0.4362 | -0.0445 | 6.93 | 6.64 | -0.29 |
| triviaqa | val | 2.5948 | 2.6066 | +0.0118 | 0.5934 | 0.5506 | -0.0428 | 13.39 | 13.55 | +0.16 |
| newsqa | train | 1.2466 | 1.2643 | +0.0177 | 0.4629 | 0.4696 | +0.0067 | 3.48 | 3.54 | +0.06 |
| newsqa | val | 2.6708 | 2.5500 | -0.1208 | 0.5465 | 0.5708 | +0.0243 | 14.45 | 12.81 | -1.64 |
| mrqa | train | 0.5731 | 0.5334 | -0.0397 | 0.2723 | 0.3064 | +0.0341 | 1.77 | 1.70 | -0.07 |
| mrqa | val | 1.2422 | 1.1976 | -0.0446 | 0.4685 | 0.4528 | -0.0157 | 3.46 | 3.31 | -0.15 |
| adversarialqa | train | 1.2722 | 1.2434 | -0.0288 | 0.4284 | 0.4229 | -0.0055 | 3.57 | 3.47 | -0.10 |
| adversarialqa | val | 3.1200 | 2.9392 | -0.1808 | 0.6351 | 0.5621 | -0.0730 | 22.65 | 18.90 | -3.75 |
| quac | train | 2.2373 | 2.0978 | -0.1395 | 0.4412 | 0.3926 | -0.0486 | 9.37 | 8.15 | -1.22 |
| quac | val | 2.5623 | 2.4692 | -0.0931 | 0.3892 | 0.4110 | +0.0218 | 12.97 | 11.81 | -1.16 |
| coqa | train | 2.5684 | 2.4571 | -0.1113 | 0.5728 | 0.6575 | +0.0847 | 13.05 | 11.67 | -1.38 |
| coqa | val | 2.6983 | 2.5015 | -0.1968 | 0.5712 | 0.5995 | +0.0283 | 14.85 | 12.20 | -2.65 |

### Per Dataset: Train-Val Gap Comparison

Gap definition: `val loss - train loss`.

| Dataset | Gap (05-31) | Gap (06-01) | Δ Gap |
|---|---:|---:|---:|
| squad_v2 | +0.8120 | +1.4270 | +0.6150 |
| natural_questions | +0.7975 | +0.6443 | -0.1532 |
| triviaqa | +0.6584 | +0.7133 | +0.0549 |
| newsqa | +1.4241 | +1.2857 | -0.1384 |
| mrqa | +0.6691 | +0.6642 | -0.0049 |
| adversarialqa | +1.8478 | +1.6958 | -0.1520 |
| quac | +0.3250 | +0.3714 | +0.0464 |
| coqa | +0.1299 | +0.0444 | -0.0855 |

### Dataset Analysis Comparison: AggAnsRate

| Dataset | AggAnsRate trn/val (05-31) | AggAnsRate trn/val (06-01) |
|---|---|---|
| squad_v2 | 0.68/0.46 | 0.68/0.46 |
| natural_questions | 0.51/0.69 | 0.51/0.69 |
| triviaqa | 1.00/1.00 | 1.00/1.00 |
| newsqa | 1.00/1.00 | 1.00/1.00 |
| mrqa | 1.00/1.00 | 1.00/1.00 |
| adversarialqa | 1.00/1.00 | 1.00/1.00 |
| quac | 0.81/0.77 | 0.83/0.78 |
| coqa | 0.99/0.99 | 0.99/0.99 |

## Notes

- Run 2 was executed today (`2026-06-01`) as requested for the fixed QnaAns dataset path after noanswer filtering.
- For metric comparison, both runs are kept with the same evaluation settings (`BATCH_SIZE=10`, `N_EVAL_BATCHES=80`, `RANDOM_SEED=42`) so the joined table is directly comparable.

## Insights And Potential Fixes (Placed At End)

### Important Observation Captured During Baseline Eval

- Baseline run note: `QnaDatasetAgg maps position indices [0..N] directly to HF row indices, meaning the noanswer-filtered sub_ds.inds is bypassed in the aggregator path.`
- This explains why `AggAnsRate` can be below `1.0` for some datasets/splits even with `exclude_noanswer=True` and cache files present.

### Ranked Flaws And Potential Fixes

#### 1) Most Important: Aggregator Indexing Bug

- Flaw: `QnaDatasetAgg` stores per-dataset local positions `0..len(ds)-1`, but forwards them directly into each child dataset `_get_item(...)` call.
- Impact: this bypasses each child dataset's filtered `inds` mapping, so `exclude_noanswer=True` can be respected during cache construction while still leaking unanswerable rows at sample-fetch time.
- Evidence: `AggAnsRate` is below `1.0` for `squad_v2`, `natural_questions`, `quac`, and `coqa` despite filtered cache files being loaded.
- Fix: resolve the aggregate-local position through `ds.inds[local_pos]` before calling the child dataset `_get_item(...)`.
- Status in this report: Run 2 is the fixed-QnaAns run requested today.

#### 2) Train/Val Composition Mismatch

- Flaw: because the aggregator path can bypass filtered indices, the effective training and validation mixtures differ from the intended answerable-only subsets.
- Impact: this creates a structural train/val mismatch and makes the observed gap larger than a normal overfitting explanation would predict.
- Fix: patch the aggregator first, then rerun the same per-dataset answerability and loss checks to confirm the effective mixture is now consistent with the cached filtered subsets.

#### 3) Dataset Difficulty Imbalance

- Flaw: a single mixed objective is dominated by easier sources such as `mrqa`, while harder sources such as `adversarialqa` and `newsqa` retain much larger validation gaps.
- Impact: optimization drifts toward easy datasets and under-serves the harder sources that dominate the overall validation loss.
- Fix: introduce source-balanced sampling or per-source loss weighting so hard datasets are represented proportionally during optimization.

#### 4) Validation Protocol Variance

- Flaw: evaluation uses `N_EVAL_BATCHES = 80` instead of a deterministic full-split pass.
- Impact: this increases variance, especially for smaller or more heterogeneous datasets such as `quac` and `coqa`, and weakens early-stopping signals.
- Fix: prefer full-split evaluation, or at minimum use a fixed source-stratified subset so repeated runs are directly comparable.

#### 5) Random Answer Selection And Long-Answer Entropy

- Flaw: some datasets, especially `natural_questions`, contain long or multiple acceptable answers, which increases target entropy if answer selection is random or unstable.
- Impact: both training and validation become noisier, and dataset comparisons become less interpretable.
- Fix: use deterministic answer selection for validation and consider stricter answer-length handling or a curriculum for very long answers.

#### 6) Mixed Single-Turn And Multi-Turn Supervision

- Flaw: single-turn QA and conversational QA are optimized together under one decoder objective despite clear prompt-format and response-distribution differences.
- Impact: this can create interference even when the data path is correct.
- Fix: consider phased training, source-aware conditioning, or a short mixed fine-tune after separate specialization phases.

#### 7) Regularization Is Secondary Here

- Flaw: treating the gap mainly as a regularization problem misses the dominant issues in data-path correctness and dataset balance.
- Impact: dropout and similar changes are unlikely to fix the main failure mode on their own.
- Fix: keep regularization tuning as a secondary step after indexing correctness and source-mixing issues are corrected.

## Training Recommendations By Dataset Hardness

### 1) Hardness Tiers

- Easy: `mrqa`, `squad_v2`
- Medium: `natural_questions`, `triviaqa`
- Hard: `adversarialqa`, `newsqa`, `quac`, `coqa`

### 2) Source-Balanced Sampling With Hardness Upweighting

- Use fixed per-source sampling quotas instead of pure dataset-size proportional sampling.
- Apply hardness multipliers and renormalize each epoch:
	- easy: `0.8`
	- medium: `1.0`
	- hard: `1.3`

### 3) Optimize For Gap Reduction, Not Only Absolute Loss

- Track per-dataset generalization gap:
	- `gap_d = val_loss_d - train_loss_d`
- Adapt source weights when gap remains high:
	- `w_d <- w_d * (1 + alpha * normalized_gap_d)`
	- recommended `alpha`: `0.1` to `0.2`

### 4) Hard-Batch Mixing Rule

- Ensure hard datasets appear in each gradient-accumulation window.
- Suggested microbatch composition target:
	- hard: `50%`
	- medium: `30%`
	- easy: `20%`

### 5) Dataset-Specific Loss Scaling

- Keep a shared model, but scale per-source loss:
	- `L = sum_d lambda_d * L_d`
- Suggested initial lambdas:
	- `lambda_adversarialqa = 1.4`
	- `lambda_newsqa = 1.3`
	- `lambda_quac = 1.2`
	- `lambda_coqa = 1.2`
	- all other datasets: `1.0`

### 6) Three-Phase Curriculum

- Phase A (stabilization): `2-3` epochs with easy+medium emphasis.
- Phase B (hardness focus): `4-6` epochs with hard upweighting.
- Phase C (re-balance): `1-2` epochs with near-uniform mixing.

### 7) Guardrails Against Overfitting To Hard Sources

- Use macro-average across datasets for early stopping.
- Add per-source patience: stop increasing a source weight if its validation loss worsens for `2` consecutive evaluations.

### 8) Reduce Evaluation Noise On Hard Sources

- Use deterministic per-source validation subsets (fixed indices).
- For long-answer sources (especially `natural_questions`), use answer-length bucketing/capping during training to reduce gradient noise.

### 9) Suggested Next Experiment

1. Enable source-balanced sampler.
2. Apply hardness-weighted loss (hard datasets only).
3. Run a short schedule (`20-30%` of full training).
4. Compare per-dataset validation loss and train-val gap against Run 2 in this report.
