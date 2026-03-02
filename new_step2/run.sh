#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  new_step2: Residual Refinement with Structure Alignment
# ═══════════════════════════════════════════════════════════════════════════
#
#  Usage:
#    bash run.sh          # train (joint mode, recommended)
#    bash run.sh eval     # evaluate best checkpoint
#
# ═══════════════════════════════════════════════════════════════════════════

MODE=${1:-train}

# ── Common paths ──────────────────────────────────────────────────────────
DATA_DIR="/root/dreamNav/pairUAV"
TRAIN_DIR="${DATA_DIR}/train/"
TEST_DIR="${DATA_DIR}/test/"
STEP1_TEST_JSON="/root/dreamNav/step1/step1_true_seen.json"
SAVE_DIR="./outputs"

if [ "$MODE" = "eval" ]; then
    echo "=== Evaluating ==="
    python evaluate.py \
        --data_dir ${DATA_DIR} \
        --test_dir ${TEST_DIR} \
        --step1_test_json ${STEP1_TEST_JSON} \
        --save_dir ${SAVE_DIR} \
        --batch_size 128 \
        --img_size 224
else
    echo "=== Training (joint mode) ==="
    python train.py \
        --data_dir ${DATA_DIR} \
        --train_dir ${TRAIN_DIR} \
        --test_dir ${TEST_DIR} \
        --step1_test_json ${STEP1_TEST_JSON} \
        --save_dir ${SAVE_DIR} \
        \
        --img_size 224 \
        --backbone resnet18 \
        --max_heading_residual_deg 45.0 \
        --max_range_residual 40.0 \
        \
        --warp_type affine_st \
        --heading_warp_scale 0.5 \
        --range_warp_scale 0.3 \
        --edge_sigma 1.5 \
        --edge_threshold 0.3 \
        \
        --mode joint \
        --lambda_pose 1.0 \
        --lambda_heading 1.0 \
        --lambda_range 1.0 \
        --lambda_struct 0.1 \
        --struct_scales 1.0 0.5 0.25 \
        --struct_warmup_epochs 5 \
        --robust_struct_loss \
        --struct_loss_clamp 5.0 \
        \
        --batch_size 64 \
        --lr 1e-4 \
        --lr_backbone 1e-5 \
        --weight_decay 1e-5 \
        --epochs 1 \
        --warmup_epochs 0 \
        --scheduler cosine \
        --grad_clip 1.0 \
        --num_workers 8 \
        --print_freq 20 \
        --quick_test_freq 1000 \
        --quick_test_samples 2000 \
        \
        --heading_noise_std 40.0 \
        --range_noise_std 30.0 \
        \
        --seed 42
fi
