import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel

NUM_CLASSES = 1000


def model_and_diffusion_defaults():
    """
    Defaults for image training. 深度学习模型，和扩散的缺省的变量字典
    """
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False, # 预测x_0
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )


def create_model_and_diffusion(
    image_size, # 图片size; 64
    class_cond, # 是否是带class条件的; True
    learn_sigma, # 是否学习sigma 方差; True
    sigma_small, # False
    num_channels, # 128
    num_res_blocks, # 3
    num_heads, # 4
    num_heads_upsample, # -1
    attention_resolutions, # 在哪些block上做attention? TODO; '16,8'
    dropout, # 0.0
    diffusion_steps, # 4000
    noise_schedule, # 'cosine'
    timestep_respacing, # ''
    use_kl, # 是否使用kl散度; True
    predict_xstart, # False
    rescale_timesteps, # True
    rescale_learned_sigmas, # True
    use_checkpoint, # False
    use_scale_shift_norm, # True
):
    import ipdb; ipdb.set_trace()
    model = create_model( # 开始创建模型
        image_size, # 64
        num_channels, # 128
        num_res_blocks, # 3
        learn_sigma=learn_sigma, # True
        class_cond=class_cond, # True，带class条件（每个图片上的物体的标签分类）
        use_checkpoint=use_checkpoint, # False
        attention_resolutions=attention_resolutions, # '16,8'
        num_heads=num_heads, # 4
        num_heads_upsample=num_heads_upsample, # -1
        use_scale_shift_norm=use_scale_shift_norm, # True
        dropout=dropout, # 0.0
    ) # TODO 创建模型，主要是对UNet的初始化，左边16层，右边16层，中间3层。

    import ipdb; ipdb.set_trace() # 开始初始化diffusion
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps, # 4000
        learn_sigma=learn_sigma, # True
        sigma_small=sigma_small, # False
        noise_schedule=noise_schedule, # 'cosine'
        use_kl=use_kl, # True use kl loss
        predict_xstart=predict_xstart, # False, 是否预测x_0，目前是不预测x_0
        rescale_timesteps=rescale_timesteps, # True
        rescale_learned_sigmas=rescale_learned_sigmas, # True
        timestep_respacing=timestep_respacing, # ''
    ) # TODO 创建"扩散过程"

    import ipdb; ipdb.set_trace()
    return model, diffusion
    # model = <class 'improved_diffusion.unet.UNetModel'>
    # diffusion = <class 'improved_diffusion.respace.SpacedDiffusion'>
def create_model(
    image_size, # 64
    num_channels, # 128
    num_res_blocks, # 3
    learn_sigma, # True
    class_cond, # True
    use_checkpoint, # False
    attention_resolutions, # '16,8'
    num_heads, # 4
    num_heads_upsample, # -1
    use_scale_shift_norm, # True
    dropout, # 0.0
):
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4) # NOTE, here for cifar10
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","): # '16,8'
        attention_ds.append(image_size // int(res)) # attention dimensions, 64/16=4, and 64/8=8, so attention_ds=[4,8]

    return UNetModel(
        in_channels=3,
        model_channels=num_channels, # 128
        out_channels=(3 if not learn_sigma else 6), # 6
        num_res_blocks=num_res_blocks, # 3
        attention_resolutions=tuple(attention_ds), # (4, 8)
        dropout=dropout, # 0.0
        channel_mult=channel_mult, # (1, 2, 3, 4)
        num_classes=(NUM_CLASSES if class_cond else None), # 1000
        use_checkpoint=use_checkpoint, # False
        num_heads=num_heads, # 4
        num_heads_upsample=num_heads_upsample, # -1
        use_scale_shift_norm=use_scale_shift_norm, # True
    )


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    _ = small_size  # hack to prevent unused variable

    if large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *,
    steps=1000, # 4000
    learn_sigma=False, # True
    sigma_small=False, # False
    noise_schedule="linear", # 'cosine'
    use_kl=False, # True
    predict_xstart=False, # False
    rescale_timesteps=False, # True
    rescale_learned_sigmas=False, # True
    timestep_respacing="", # ''
):
    # 生成扩散的框架
    import ipdb; ipdb.set_trace()
    betas = gd.get_named_beta_schedule(noise_schedule, steps) # determine beta schedule

    if use_kl: # True
        loss_type = gd.LossType.RESCALED_KL # <LossType.RESCALED_KL: 4> NOTE, here
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE # 这就是和原来的ddpm文章一样，直接使用的是ddpm的mse loss
    if not timestep_respacing:
        timestep_respacing = [steps] # NOTE, here, [4000]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing), # {0, 1, ..., 3999}
        betas=betas, # [9.86581881e-06, 1.01694149e-05, 1.04730140e-05, ..., 0.74999996, 0.999]
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ), # <ModelMeanType.EPSILON: 3>
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE # <ModelVarType.LEARNED_RANGE: 4>
        ),
        loss_type=loss_type, # <LossType.RESCALED_KL: 4>
        rescale_timesteps=rescale_timesteps, # True
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)
        # 这是从字典中，自动生产出来arg parser，循环地搞起add_argument了！NOTE

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}
    # args: 大的超参数的集合； Namespace(attention_resolutions='16,8', batch_size=128, class_cond=True, data_dir='/workspace/asr/diffusion_models/improved-diffusion/datasets/cifar_train', diffusion_steps=4000, dropout=0.0, ema_rate='0.9999', fp16_scale_growth=0.001, image_size=64, learn_sigma=True, log_interval=10, lr=0.0001, lr_anneal_steps=0, microbatch=-1, noise_schedule='cosine', num_channels=128, num_heads=4, num_heads_upsample=-1, num_res_blocks=3, predict_xstart=False, rescale_learned_sigmas=True, rescale_timesteps=True, resume_checkpoint='', save_interval=10000, schedule_sampler='loss-second-moment', sigma_small=False, timestep_respacing='', use_checkpoint=False, use_fp16=False, use_kl=True, use_scale_shift_norm=True, weight_decay=0.0); TODO len(keys)=19, -> dict_keys(['image_size', 'num_channels', 'num_res_blocks', 'num_heads', 'num_heads_upsample', 'attention_resolutions', 'dropout', 'learn_sigma', 'sigma_small', 'class_cond', 'diffusion_steps', 'noise_schedule', 'timestep_respacing', 'use_kl', 'predict_xstart', 'rescale_timesteps', 'rescale_learned_sigmas', 'use_checkpoint', 'use_scale_shift_norm']) NOTE return={'attention_resolutions': '16,8', 'class_cond': True, 'diffusion_steps': 4000, 'dropout': 0.0, ...}

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
