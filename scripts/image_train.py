"""
Train a diffusion model on images.
"""
import os
import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop

import numpy as np
import random
import torch

def set_rand_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

def main():
    import ipdb; ipdb.set_trace()
    # fix random seeds
    os.environ["OPENAI_LOGDIR"] = "/workspace/asr/diffusion_models/improved-diffusion/checkpoints"

    seed = 666
    set_rand_seed(666)

    args = create_argparser().parse_args() # 各种控制参数的“汇总” NOTE step 1, type(args) = <class 'argparse.Namespace'>

    dist_util.setup_dist() # 分布式计算的一些配置，目前不看细节 TODO
    logger.configure() # Logging to /tmp/openai-2023-01-07-23-30-56-035823 -> /workspace/asr/diffusion_models/improved-diffusion/checkpoints

    import ipdb; ipdb.set_trace()
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()) # NOTE` 创建model和diffusion对象 step 2
    )
    model.to(dist_util.dev()) # device(type='cuda', index=0)

    import ipdb; ipdb.set_trace()
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion) # TODO 什么含义? 返回的是采样器; args.schedule_sampler='loss-second-moment', duffusion=一个具体的object = <improved_diffusion.respace.SpacedDiffusion object at 0x7f4f51296250>

    import ipdb; ipdb.set_trace()
    logger.log("creating data loader...") # step 3, load data，导入数据
    data = load_data(
        data_dir=args.data_dir, # '/workspace/asr/diffusion_models/improved-diffusion/datasets/cifar_train'
        batch_size=args.batch_size, # 128
        image_size=args.image_size, # 64
        class_cond=args.class_cond, # True
    )

    logger.log("training...") # step 4, 开启训练loop
    import ipdb; ipdb.set_trace()
    TrainLoop(
        model=model, # <class 'improved_diffusion.unet.UNetModel'>
        diffusion=diffusion, # <improved_diffusion.respace.SpacedDiffusion object at 0x7f4cbc369f70>
        data=data, # <generator object load_data at 0x7f4cbc3237b0>
        batch_size=args.batch_size, # 128
        microbatch=args.microbatch, # -1
        lr=args.lr, # 0.0001
        ema_rate=args.ema_rate, # '0.9999'
        log_interval=args.log_interval, # 10
        save_interval=args.save_interval, # 10000
        resume_checkpoint=args.resume_checkpoint, # ''
        use_fp16=args.use_fp16, # False
        fp16_scale_growth=args.fp16_scale_growth, # 0.001
        schedule_sampler=schedule_sampler, # <improved_diffusion.resample.LossSecondMomentResampler object at 0x7f4db0788490>
        weight_decay=args.weight_decay, # 0.0
        lr_anneal_steps=args.lr_anneal_steps, # None
    ).run_loop()

    import ipdb; ipdb.set_trace()
    # finally stop

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser() # 接受的来自命令行的参数key-value
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
