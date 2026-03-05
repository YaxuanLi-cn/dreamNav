#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

DATASET_PATH="../pairUAV"
TRAIN_PATH="${DATASET_PATH}/train/"
TEST_PATH="${DATASET_PATH}/test/"

python step2.py \
    --data_dir "$DATASET_PATH" \
    --train_path "$TRAIN_PATH" \
    --test_path "$TEST_PATH" \
    --test_max_samples 1000 \
    --img_size 256 \
    --hidden_dim 256 \
    --batch_size 256 \
    --num_workers 8 \
    --epochs 10 \
    --lr 1e-3 \
    --print_freq 100 \
    --vis_freq 2000 \
    --vis_dir vis_output \
    --save_path step2_model.pth
