"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2): # mean1 and logvar1 for reference (true distribution's mean and logvar)
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    ) # 两个正态分布的KL散度, shape=[1, 3, 64, 64]


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal. 标准正态分布的 累计分布函数 的一个快速近似:
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))
    # x.shape=[1, 3, 64, 64]; out.shape=[1, 3, 64, 64]
# NOTE 这个方法完全是新知识：TODO
def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1]. 观测到的数据，x_start.shape=[1, 3, 64, 64]。
    :param means: the Gaussian mean Tensor. 均值向量, means.shape=[1, 3, 64, 64].
    :param log_scales: the Gaussian log stddev Tensor. 标准方差的对数, log_scales=[1, 3, 64, 64]。
    :return: a tensor like x of log probabilities (in nats).
    """
    # 连续的累计高斯分布函数的很小的两个值的差分，来模拟，离散的高斯分布的概率，NOTE
    import ipdb; ipdb.set_trace() # NOTE
    assert x.shape == means.shape == log_scales.shape
    # 减去均值: 
    centered_x = x - means # [1, 3, 64, 64]
    inv_stdv = th.exp(-log_scales) # 标准方差的倒数, [1, 3, 64, 64]

    # 将[-1, 1]分成255个bins，最右边的cdf记为1，最左边的cdf记为0:
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0) # [1, 3, 64, 64]
    cdf_plus = approx_standard_normal_cdf(plus_in) # cdf = 累计分布函数；这是去计算一个标准分布的，累计分布函数. NOTE, cdf_plus.shape=[1, 3, 64, 64]

    min_in = inv_stdv * (centered_x - 1.0 / 255.0) # [1, 3, 64, 64]
    cdf_min = approx_standard_normal_cdf(min_in) # [1, 3, 64, 64]

    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12)) # 稳定性，确保不能为0; 这是左边的一个极限值, [1. 3, 64, 64]
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12)) # 两个辅助的遍历, 这是右边的一个极限值, [1, 3, 64, 64]

    # 用小范围的cdf之差来表示pdf, TODO
    cdf_delta = cdf_plus - cdf_min # [1, 3, 64, 64]

    # 对数概率：考虑到两个极限的地方，这里用到了两个where
    log_probs = th.where(
        x < -0.999, # x在最左边的时候，
        log_cdf_plus, # 取左边的极限值
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))), # x在最右边的时候，取右边的极限值。如果不在两个极限的附近，则取值是最后一个th.log，正常取log，即可。
    )
    assert log_probs.shape == x.shape
    return log_probs # 最后返回的是，对数似然, log_probs.shape=[1, 3, 64, 64]
