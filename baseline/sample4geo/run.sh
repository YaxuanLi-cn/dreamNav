#!/bin/bash

EMBEDDING_DIR="/root/dreamNav/baseline/sample4geo/embedding"
TRAIN_DIR="/root/dreamNav/pairUAV/train"
TEST_DIR="/root/dreamNav/pairUAV/test"

OUTPUT_FILE="test_results.log"

python train.py \
    --embedding_dir ${EMBEDDING_DIR} \
    --train_dir ${TRAIN_DIR} \
    --test_dir ${TEST_DIR} \
    --epochs 10 \
    --lr_regressor 1e-3 \
    --warmup_epochs 1 \
    --output_file ${OUTPUT_FILE}
