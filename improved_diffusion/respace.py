import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up. e.g., num_timesteps=4000
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper. e.g., section_counts=[4000]
    :return: a set of diffusion steps from the original process to use.
    """
    import ipdb; ipdb.set_trace()
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts) # 4000 / 1 = 4000 = size_per
    extra = num_timesteps % len(section_counts) # 0
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts): # i=0, section_count=4000
        size = size_per + (1 if i < extra else 0)
        if size < section_count: # size=4000, section_count=4000
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1) # frac_stride=1.0
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps # all_steps=[0, 1, ..., 3999]
        start_idx += size # start_idx=4000
    return set(all_steps) # 这是变成一个集合了, {0, 1, ..., 3999}


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs): # dict_keys(['betas', 'model_mean_type'=epsilon:3, 'model_var_type=learned_range:4', 'loss_type:rescaled_kl:4', 'rescale_timesteps=True']) for 'kwargs'!
        import ipdb; ipdb.set_trace()
        self.use_timesteps = set(use_timesteps) # {0, 1, ..., 3999}
        self.timestep_map = [] # NOTE, 是连续的，还是跳跃式的(spaced的)
        self.original_num_steps = len(kwargs["betas"]) # 做多少步的加噪, 4000
        import ipdb; ipdb.set_trace() # 
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa; NOTE 这是直接调用父类的构造函数...
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps: # 4000 for 4000, 目前都在里面
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas) # 对betas求和=15.210434270813122; 对new_betas求和：15.210434270813133; 两者基本没有变化... NOTE
        super().__init__(**kwargs) # NOTE, why, call __init__ again? Line 78已经call过了啊... 主要原因：因为betas -> new_betas了，有可能修改了betas，所以这里重新计算了从betas, alphas，到其他所有的一共14个变量。NOTE okay.

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs 
        import ipdb; ipdb.set_trace()
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)
        # model is already "_WrappedModel", so just return 'model'
    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        # NOTE, 根据传入的loss type的不同，得到不同的损失函数
        import ipdb; ipdb.set_trace()
        # kl loss, mse loss, and so on
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        ) # self.rescale_timesteps=True, self.original_num_steps=4000

    def _scale_timesteps(self, t): # e.g., t=tensor([2801], device='cuda:0')
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        import ipdb; ipdb.set_trace()
        self.model = model # <class 'improved_diffusion.unet.UNetModel'>
        self.timestep_map = timestep_map # a list with 4000 values: [0, 1, 2, 3, 4, ...]
        self.rescale_timesteps = rescale_timesteps # True
        self.original_num_steps = original_num_steps # 4000

    def __call__(self, x, ts, **kwargs): # x.shape=[1, 3, 64, 64], ts=[2801], kwargs={'y': tensor([1], device='cuda:0')}
        import ipdb; ipdb.set_trace()
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype) # tensor([   0,    1,    2,  ..., 3997, 3998, 3999], device='cuda:0')
        new_ts = map_tensor[ts] # tensor([2801], device='cuda:0')
        if self.rescale_timesteps: # True NOTE
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps) # 2801 * 1000.0/4000 = tensor([700.2500], device='cuda:0')

        import ipdb; ipdb.set_trace()
        return self.model(x, new_ts, **kwargs) # NOTE forward
        # x.shape = [1, 3, 64, 64]
        # new_ts = tensor([700.2500], device='cuda:0') 对时间缩放，从0-4000到0-1000了。
        # kwargs = {'y': tensor([1], device='cuda:0')}
        
        # out.shape = [1, 6, 64, 64], first 3 for mean and next 3 for variance
