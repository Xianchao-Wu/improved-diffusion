#########################################################################
# File Name: run_train.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Sat Jan  7 06:59:37 2023
#########################################################################
#!/bin/bash

MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True" # cosine or linear
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"

python -m ipdb scripts/image_train.py \
    --data_dir /workspace/asr/diffusion_models/improved-diffusion/datasets/cifar_train \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $TRAIN_FLAGS


