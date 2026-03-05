#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

python step2_test.py \
    --step1_json "../step1/step1_seen.json" \
    --data_dir "../pairUAV" \
    --model_path "before.pth" \
    --img_size 256 \
    --hidden_dim 256 \
    --range_delta 0 \
    --heading_delta 10 \
    --max_samples 10000
