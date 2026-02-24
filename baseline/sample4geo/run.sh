#!/bin/bash

IMAGE_ROOT="/root/dreamNav/pairUAV/tours"
TRAIN_DIR="/root/dreamNav/pairUAV/train"
TEST_DIR="/root/dreamNav/pairUAV/test"
CHECKPOINT="pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth"

OUTPUT_FILE="test_results_e2e.log"

python train.py \
    --image_root ${IMAGE_ROOT} \
    --train_dir ${TRAIN_DIR} \
    --test_dir ${TEST_DIR} \
    --checkpoint ${CHECKPOINT} \
    --model_name convnext_base.fb_in22k_ft_in1k_384 \
    --img_size 384 \
    --batch_size 5 \
    --epochs 1 \
    --lr_backbone 1e-5 \
    --lr_regressor 1e-3 \
    --warmup_epochs 0 \
    --output_file ${OUTPUT_FILE}
