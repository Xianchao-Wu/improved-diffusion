#########################################################################
# File Name: run_train.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Sat Jan  7 06:59:37 2023
#########################################################################
#!/bin/bash

MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True" # cosine or linear
TRAIN_FLAGS="--lr 1e-4 --batch_size 16 --schedule_sampler loss-second-moment"
# batch_size=128 or 1 for debug only

python -m ipdb scripts/image_train.py \
    --data_dir /workspace/asr/diffusion_models/improved-diffusion/datasets/cifar_train_debug \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $TRAIN_FLAGS


#ipdb>
#usage: image_train.py [-h] [--data_dir DATA_DIR] [--schedule_sampler SCHEDULE_SAMPLER] [--lr LR]
#                      [--weight_decay WEIGHT_DECAY] [--lr_anneal_steps LR_ANNEAL_STEPS] [--batch_size BATCH_SIZE]
#                      [--microbatch MICROBATCH] [--ema_rate EMA_RATE] [--log_interval LOG_INTERVAL]
#                      [--save_interval SAVE_INTERVAL] [--resume_checkpoint RESUME_CHECKPOINT] [--use_fp16 USE_FP16]
#                      [--fp16_scale_growth FP16_SCALE_GROWTH] [--image_size IMAGE_SIZE] [--num_channels NUM_CHANNELS]
#                      [--num_res_blocks NUM_RES_BLOCKS] [--num_heads NUM_HEADS] [--num_heads_upsample NUM_HEADS_UPSAMPLE]
#                      [--attention_resolutions ATTENTION_RESOLUTIONS] [--dropout DROPOUT] [--learn_sigma LEARN_SIGMA]
#                      [--sigma_small SIGMA_SMALL] [--class_cond CLASS_COND] [--diffusion_steps DIFFUSION_STEPS]
#                      [--noise_schedule NOISE_SCHEDULE] [--timestep_respacing TIMESTEP_RESPACING] [--use_kl USE_KL]
#                      [--predict_xstart PREDICT_XSTART] [--rescale_timesteps RESCALE_TIMESTEPS]
#                      [--rescale_learned_sigmas RESCALE_LEARNED_SIGMAS] [--use_checkpoint USE_CHECKPOINT]
#                      [--use_scale_shift_norm USE_SCALE_SHIFT_NORM]
#image_train.py: error: argument --learn_sigma: expected one argument

