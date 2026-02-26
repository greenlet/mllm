# Copy train directories from Cosmos to local
# Usage: .\copy_encdec_train_from_cosmos.sh [-override]

param(
    [switch]$override
)

$root_src = "https://coesmos11.osdinfra.net:443/cosmos/SegmentRelevance/local/users/mburakov/data/train_mllm_encdec_bert"
# $root_src = "vc://cosmos11/SegmentRelevance/local/users/mburakov/data/train_mllm_encdec_bert"
$root_dst = "Q:/data/train_mllm_encdec_bert"

$dirs = @(
    "encdecbert-20260110_193915-bertbaseuncased-d768-embCls-inp128-lrs7x1-enhMmbb-step2-h12-dp0-t0.0"
    "encdecgraphbert-20260206_210244-bertbaseuncased-d768-embCls-inp128-enhMmbb-lrs7x1-step2-h12-swF-dp0-embmlp_win3_wlrs6_olrs2_actGelu-trn_ctokAll_scl1.0_w1.0_itok_scl1.0_w1.0_lr5e-05_bs40"
    "encdecgraphbert-20260208_033457-pre_encdecbert20260110193915-bertbaseuncased-d768-embCls-inp128-enhMmbb-lrs7x1-step2-h12-swF-dp0-embmlp_win3_wlrs6_olrs2_actGelu-trn_ctokAll_scl1.0_w1.0_cembCosl2_scl1.0_w1.0_itok_scl1.0_w1.0_lr5e-05_bs60"
    "encdecgraphbert-20260213_121906-pre_encdecbert20260110193915-bertbaseuncased-d768-embCls-inp128-enhMmbb-lrs7x1-step2-h12-swF-dp0-embcross_win3_h8_lrs2_dp0.1_gmlpT-trn_ctokAll_scl1.0_w1.0_cembCosl2_scl1.0_w1.0_itok_scl1.0_w1.0_lr5e-05_bs60"
    "encdecgraphbert-20260216_094353-pre_encdecbert20260110193915-bertbaseuncased-d768-embCls-inp128-enhMmbb-lrs7x1-step2-h12-swF-dp0-embcross_win3_h8_lrs3_dp0.1_gmlpT-trn_ctokAll_scl1.0_w1.0_cembCosl2_scl1.0_w1.0_itok_scl1.0_w1.0_lr5e-05_bs60"
    "encdecgraphbert-20260219_214345-pre_encdecbert20260110193915-bertbaseuncased-d768-embCls-inp128-enhMmbb-lrs7x1-step2-h12-swF-dp0-embcross_win3_h8_lrs2_dp0.1_gmlpF_exp4-trn_ctokAll_scl1.0_w1.0_cembCosl2_scl1.0_w1.0_itok_scl1.0_w1.0_lr5e-05_bs50"
    "encdecgraphbert-20260220_143251-pre_encdecbert20260110193915-bertbaseuncased-d768-embCls-inp128-enhMmbb-lrs7x1-step2-h12-swF-dp0-embcross_win3_h8_lrs4_dp0.1_gmlpF_exp4-trn_ctokAll_scl1.0_w1.0_cembCosl2_scl1.0_w1.0_itok_scl1.0_w1.0_lr5e-05_bs50"
    "encdecgraphbert-20260223_203433-pre_encdecbert20260110193915-bertbaseuncased-d768-embCls-inp128-enhMmbb-lrs7x1-step2-h12-swF-dp0-embcross_win3_h8_lrs2_dp0.1_gmlpT-trn_ctokAll_scl1.0_w1.0_cembCosl2_scl1.0_w1.0_itok_scl1.0_w1.0_lr5e-05_bs50"
    "encdecgraphbert-20260224_215434-pre_encdecbert20260110193915-bertbaseuncased-d768-embCls-inp128-enhMmbb-lrs7x1-step2-h12-swF-dp0-embgate_exp4_dp0.1-trn_ctokAll_scl1.0_w1.0_cembCosl2_scl1.0_w1.0_itok_scl1.0_w1.0_lr5e-05_bs50"
    "encdecgraphbert-20260227_122831-pre_encdecbert20260110193915-bertbaseuncased-d768-embCls-inp128-enhMmbb-lrs7x1-step2-h12-swF-dp0-embgate_exp4_nl8_dp0.1-trn_ctokAll_scl1.0_w1.0_cembCosl2_scl1.0_w1.0_itok_scl1.0_w1.0_lr5e-05_bs50"
)

foreach ($dir in $dirs) {
    $dir_src = "$root_src/$dir"
    $dir_dst = "$root_dst/$dir"

    if (-not $override -and (Test-Path $dir_dst)) {
        Write-Host "Skip existing: $dir"
        continue
    }
    # Write-Host "$dir_src -> $dir_dst"
    Write-Host "Download: $dir"
    New-Item -ItemType Directory -Force -Path $dir_dst | Out-Null
    Export-CosmosStreamToFile -Recurse -Overwrite $dir_src $dir_dst
}

