#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

MODEL_PATH="../models/dino_resnet"
SUPERGLUE_OUTPUT="matches_data/"
DATASET_PATH="../pairUAV"
TRAIN_PATH="${DATASET_PATH}/train/"
TEST_PATH="${DATASET_PATH}/test/"
NUM_WORKERS=64
BATCH_SIZE=256


python step1_no_match.py \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --data_dir "$DATASET_PATH" \
    --model_path "$MODEL_PATH" \
    --train_path "$TRAIN_PATH" \
    --test_path "$TEST_PATH" \
    --epochs 5 \
    --warmup_epochs 1 \
    --lr 1e-6 \
    --lr_regressor 5e-3 \
    --momentum 0.9 \
    --wd 1e-10 \
    --print_freq 10 \
    --seed 2021


python step1_ablation_norm_angle.py \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --data_dir "$DATASET_PATH" \
    --model_path "$MODEL_PATH" \
    --train_path "$TRAIN_PATH" \
    --test_path "$TEST_PATH" \
    --epochs 5 \
    --warmup_epochs 1 \
    --lr 1e-6 \
    --lr_regressor 5e-3 \
    --momentum 0.9 \
    --wd 1e-10 \
    --print_freq 10 \
    --seed 2021
