#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

python step2_test_heading.py \
    --step1_json "../step1/step1_seen.json" \
    --data_dir "../pairUAV" \
    --model_path "before.pth" \
    --candidate_file "range_num.txt" \
    --img_size 256 \
    --hidden_dim 256 \
    --num_candidates 3 \
    --max_samples 10000
