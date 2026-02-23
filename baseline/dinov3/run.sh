#!/bin/bash

TOURS_DIR="/root/dreamNav/pairUAV/tours"
MODEL_NAME="/root/dreamNav/models/dinov3_7b"
TRAIN_DIR="/root/dreamNav/pairUAV/train"
TEST_DIR="/root/dreamNav/pairUAV/test"

OUTPUT_FILE="test_results.log"

python train.py \
    --tours_dir ${TOURS_DIR} \
    --model_name ${MODEL_NAME} \
    --train_dir ${TRAIN_DIR} \
    --test_dir ${TEST_DIR} \
    --epochs 1 \
    --batch_size 16 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lr_lora 1e-4 \
    --lr_regressor 1e-3 \
    --warmup_epochs 0 \
    --output_file ${OUTPUT_FILE}
