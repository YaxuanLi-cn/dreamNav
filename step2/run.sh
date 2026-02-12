#!/bin/bash

python train.py \
    --checkpoint_path ../models/controlnet/control_sd15_ini.ckpt \
    --root_dir ../pairUAV/ \
    --dataset_type train \
    --step1_json_path ../step1/step1_seen.json \
    --test_save_dir ./outputs/test_results/ \
    --output_dir ./outputs/ \
    --batch_size 512 \
    --test_batch_size 64 \
    --logger_freq 300 \
    --learning_rate 1e-5 \
    --heading_offset 10.0 \
    --range_offset 1.5 \
    --test_ddim_steps 50 \
    --max_test_samples 6000 \
    --range_offset 1.5 \
    --heading_offset 10.0 \
    --max_epochs 5 
