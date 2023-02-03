from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb):
        #import ipdb; ipdb.set_trace() # for each block in input-block, middle-block and output-block! in unet.
        # x = x_t
        # emb = time embed or condition embed
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb) # resblock, 只需要resblock的时候，传入time embedding，其他的不需要 NOTE
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels, # 128
        emb_channels, # 512
        dropout, # 0.0
        out_channels=None, # 128
        use_conv=False, # False
        use_scale_shift_norm=False, # True
        dims=2, # 2
        use_checkpoint=False, # False
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels # 128
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels), # GroupNorm32(32, channels); GroupNorm32(32, 128, eps=1e-05, affine=True) NOTE
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        '''
        Sequential(
          (0): GroupNorm32(32, 128, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        '''

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        '''
        Sequential(
          (0): SiLU()
          (1): Linear(in_features=512, out_features=256, bias=True)
        )
        '''

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )
        '''
        Sequential(
          (0): GroupNorm32(32, 128, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Dropout(p=0.0, inplace=False)
          (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        '''

        if self.out_channels == channels:
            self.skip_connection = nn.Identity() # NOTE, here, Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        '''
        ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 128, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=512, out_features=256, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 128, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0.0, inplace=False)
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Identity()
        )
        '''

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings. NOTE 这个需要time embedding
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint # NOTE need time embedding
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        # channels=384, num_heads=4, use_checkpoint=False

        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels) # GroupNorm32(32, 384, eps=1e-05, affine=True)
        self.qkv = conv_nd(1, channels, channels * 3, 1) # Conv1d(384, 1152, kernel_size=(1,), stride=(1,)) NOTE 这个的实现代码，有意思
        self.attention = QKVAttention() # NOTE 这个没有自己的构造函数. 
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))
        '''Conv1d(384, 384, kernel_size=(1,), stride=(1,))'''

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x)) # Q, K, V
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)

        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)
        # 带残差的注意力机制，标准的


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels, # 3
        model_channels, # 128
        out_channels, # 6
        num_res_blocks, # 3
        attention_resolutions, # (4, 8)
        dropout=0, # 0.0
        channel_mult=(1, 2, 4, 8), # (1, 2, 3, 4)
        conv_resample=True, # True
        dims=2, # 2
        num_classes=None, # 1000
        use_checkpoint=False, # False
        num_heads=1, # 4
        num_heads_upsample=-1, # -1
        use_scale_shift_norm=False, # True
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads # 4

        self.in_channels = in_channels # 3
        self.model_channels = model_channels # 128
        self.out_channels = out_channels # 6
        self.num_res_blocks = num_res_blocks # 3
        self.attention_resolutions = attention_resolutions # (4,8)
        self.dropout = dropout # 0.0
        self.channel_mult = channel_mult # (1, 2, 3, 4)
        self.conv_resample = conv_resample # True
        self.num_classes = num_classes # 1000
        self.use_checkpoint = use_checkpoint # False
        self.num_heads = num_heads # 4
        self.num_heads_upsample = num_heads_upsample # -1

        time_embed_dim = model_channels * 4 # 128 * 4 = 512
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim), # model_channels 128 -> time_embed_dim 512
            SiLU(),
            linear(time_embed_dim, time_embed_dim), # time_embed_dim 512 -> time_embed_dim 512
        )
        ''' 对时间t的embedding: 
        Sequential(
          (0): Linear(in_features=128, out_features=512, bias=True)
          (1): SiLU()
          (2): Linear(in_features=512, out_features=512, bias=True)
        )
        '''

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim) # (1000, 512) 这是对1000个标签进行embeddig matrix的操作. 
        '''对分类标签的embedding: Embedding(1000, 512)'''

        self.input_blocks = nn.ModuleList( # Unet的左边的那部分
            [
                TimestepEmbedSequential( # 这个TimestepEmbedSequential的构造函数，啥也没有 NOTE
                    conv_nd(dims, in_channels, model_channels, 3, padding=1) # kernel-size=3,
                )
            ] # 列表
        )
        ''' 初始转换图片的操作，从channel = 3 to 128
        ModuleList(
          (0): TimestepEmbedSequential(
            (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        '''

        input_block_chans = [model_channels] # [128]
        ch = model_channels # 128
        ds = 1
        for level, mult in enumerate(channel_mult): # (1, 2, 3, 4)
            for _ in range(num_res_blocks): # 3; 算下来就是4*3 = 12层
                layers = [
                    ResBlock( # 1
                        ch, # 128, |||, 
                        time_embed_dim, # time embed, 512
                        dropout,
                        out_channels=mult * model_channels, # 通道数目在扩大 128, |||, 
                        dims=dims, # 2 ||| 
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions: # (4, 8); ds = down sampling，下采样的比例
                    layers.append(
                        AttentionBlock( # NOTE attention block
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims)) # down sample 层，下采样
                )
                input_block_chans.append(ch)
                ds *= 2

        '''
        起头：    (0): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        level=0, mult=1; ds=1 [4]
            0-th
               一个res block: in_layers, emb_layers, out_layers
            1-th
               一个res block: in_layers, emb_layers, out_layers
            2-th
               一个res block: in_layers, emb_layers, out_layers
            追加一个down sample

        level=1, mult=2; ds=2 [4]
            0-th
               一个res block: in_layers, emb_layers, out_layers
            1-th
               一个res block: in_layers, emb_layers, out_layers
            2-th
               一个res block: in_layers, emb_layers, out_layers
            追加一个down sample

        level=2, mult=3; ds=4 [4]
            0-th
               一个res block: in_layers, emb_layers, out_layers; 后面追加一个attention block
            1-th
               一个res block: in_layers, emb_layers, out_layers; 后面追加一个attention block
            2-th
               一个res block: in_layers, emb_layers, out_layers; 后面追加一个attention block
            追加一个down sample

        level=3, mult=4; ds=8 [3]
            0-th
               一个res block: in_layers, emb_layers, out_layers; 后面追加一个attention block
            1-th
               一个res block: in_layers, emb_layers, out_layers; 后面追加一个attention block
            2-th
               一个res block: in_layers, emb_layers, out_layers; 后面追加一个attention block

        from 0-th to 15-th, which are 16 blocks in total
        '''

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        '''
        一个resblock
        一个attention block
        一个resblock
        '''

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]: # 3,4; 2,3; 1,2; 0,1
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(), # NOTE 左边的通道数，左右有直连
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        '''
        level=3, mult=4, ds=8, i=0,1,2,3
            i=0,
                res block, attention block
            i=1,
                res block, attention block
            i=2,
                res block, attention block
            i=3,
                res block, attention block, NOTE up sample 增加一个上采样，注意这个上采样的位置！

        level=2, mult=3, ds=4, i=0,1,2,3
            i=0,
                res block, attention block
            i=1,
                res block, attention block
            i=2,
                res block, attention block
            i=3,
                res block, attention block, NOTE up sample 增加一个上采样，注意这个上采样的位置！

        level=1, mult=2, ds=2, i=0,1,2,3
            i=0,
                res block
            i=1,
                res block
            i=2,
                res block
            i=3,
                res block, NOTE up sample 增加一个上采样，注意这个上采样的位置！

        level=0, mult=1, ds=1, i=0,1,2,3
            (12) i=0,
                res block
            (13) i=1,
                res block
            (14) i=2,
                res block
            (15) i=3,
                res block
        '''

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        '''
        Sequential(
          (0): GroupNorm32(32, 128, eps=1e-05, affine=True)
          (1): SiLU()
          (2): Conv2d(128, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        '''

        ''' 整体：
        1 time_embed
        2 label_embed
        3 input_blocks
        4 middle_block
        5 output_blocks
        6 out
        '''

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs. e.g., x.shape=[1, 3, 64, 64]
        :param timesteps: a 1-D batch of timesteps. e.g., timesteps=tensor([700.2500], device='cuda:0')
        :param y: an [N] Tensor of labels, if class-conditional. e.g., y=tensor([1], device='cuda:0')
        :return: an [N x C x ...] Tensor of outputs.
        """
        import ipdb; ipdb.set_trace()
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = [] # hidden layer output tensors, 保存input_blocks的各个block的输出
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)) # timesteps=700.25, self.model_channels=128

        if self.num_classes is not None: # 1000, in
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y) # [1, 512], 重要，这是对class label做embedding! NOTE, emb.shape=[1, 512]

        h = x.type(self.inner_dtype) # h.shape=[1, 3, 64, 64], self.inner_dtype=torch.float32, 该方法的功能是: 当不指定dtype时,返回类型. 当指定dtype时 NOTE (目前在这里),返回类型转换后的数据,如果类型已经符合要求, 那么不做额外的复制,返回原对象. 

        import ipdb; ipdb.set_trace()
        for module in self.input_blocks: # 16个blocks
            h = module(h, emb) # NOTE, 需要注意的是，这里的emb，已经包括了time.emb + class.emb!
            hs.append(h)

        # 0-th, [1, 3, 64, 64], [1, 512] -> [1, 128, 64, 64] # NOTE
        # 1-th, (torch.Size([1, 128, 64, 64]), torch.Size([1, 512])) -> torch.Size([1, 128, 64, 64])
        # 2-th, (torch.Size([1, 128, 64, 64]), torch.Size([1, 512])) -> torch.Size([1, 128, 64, 64])
        # 3-th, (torch.Size([1, 128, 64, 64]), torch.Size([1, 512])) -> torch.Size([1, 128, 64, 64])

        # 4-th, (torch.Size([1, 128, 64, 64]), torch.Size([1, 512])) -> torch.Size([1, 128, 32, 32]) # NOTE
        # 5-th, (torch.Size([1, 128, 32, 32]), torch.Size([1, 512])) -> torch.Size([1, 256, 32, 32]) # NOTE
        # 6-th, (torch.Size([1, 256, 32, 32]), torch.Size([1, 512])) -> torch.Size([1, 256, 32, 32])
        # 7-th, (torch.Size([1, 256, 32, 32]), torch.Size([1, 512])) -> torch.Size([1, 256, 32, 32])

        # 8-th, (torch.Size([1, 256, 32, 32]), torch.Size([1, 512])) -> torch.Size([1, 256, 16, 16]) # NOTE
        # 9-th, (torch.Size([1, 256, 16, 16]), torch.Size([1, 512])) -> torch.Size([1, 384, 16, 16]) # NOTE 
        # 10-th, (torch.Size([1, 384, 16, 16]), torch.Size([1, 512])) -> torch.Size([1, 384, 16, 16]) 
        # 11-th, (torch.Size([1, 384, 16, 16]), torch.Size([1, 512])) -> torch.Size([1, 384, 16, 16]) 

        # 12-th, (torch.Size([1, 384, 16, 16]), torch.Size([1, 512])) -> torch.Size([1, 384, 8, 8]) # NOTE
        # 13-th, (torch.Size([1, 384, 8, 8]), torch.Size([1, 512])) -> torch.Size([1, 512, 8, 8]) # NOTE
        # 14-th, (torch.Size([1, 512, 8, 8]), torch.Size([1, 512])) -> torch.Size([1, 512, 8, 8]) 
        # 15-th, (torch.Size([1, 512, 8, 8]), torch.Size([1, 512])) -> torch.Size([1, 512, 8, 8]) 

        import ipdb; ipdb.set_trace()
        h = self.middle_block(h, emb)
        # (torch.Size([1, 512, 8, 8]), torch.Size([1, 512])) -> torch.Size([1, 512, 8, 8]) 
        import ipdb; ipdb.set_trace()
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        # 0-th, h=[1, 512, 8, 8], cat_in=[1, 1024, 8, 8], out.h=[1, 512, 8, 8]
        # 1-th, h=[1, 512, 8, 8], cat_in=[1, 1024, 8, 8], out.h=[1, 512, 8, 8]
        # 2-th, h=[1, 512, 8, 8], cat_in=[1, 1024, 8, 8], out.h=[1, 512, 8, 8]
        
        # 3-th, h=[1, 512, 8, 8], cat_in=[1, 896, 8, 8], out.h=[1, 512, 16, 16] NOTE
        # 4-th, h=[1, 512, 16, 16], cat_in=[1, 896, 16, 16], out.h=[1, 384, 16, 16] NOTE
        # 5-th, h=[1, 384, 16, 16], cat_in=[1, 768, 16, 16], out.h=[1, 384, 16, 16] NOTE
        # 6-th, h=[1, 384, 16, 16], cat_in=[1, 768, 16, 16], out.h=[1, 384, 16, 16] 

        # 7-th, h=[1, 384, 16, 16], cat_in=[1, 640, 16, 16], out.h=[1, 384, 32, 32] 
        # 8-th, h=[1, 384, 32, 32], cat_in=[1, 640, 32, 32], out.h=[1, 256, 32, 32] 
        # 9-th, h=[1, 256, 32, 32], cat_in=[1, 512, 32, 32], out.h=[1, 256, 32, 32] 
        # 10-th, h=[1, 256, 32, 32], cat_in=[1, 512, 32, 32], out.h=[1, 256, 32, 32] 

        # 11-th, h=[1, 256, 32, 32], cat_in=[1, 384, 32, 32], out.h=[1, 256, 64, 64] NOTE 
        # 12-th, h=[1, 256, 64, 64], cat_in=[1, 384, 64, 64], out.h=[1, 128, 64, 64] NOTE 
        # 13-th, h=[1, 128, 64, 64], cat_in=[1, 256, 64, 64], out.h=[1, 128, 64, 64]  
        # 14-th, h=[1, 128, 64, 64], cat_in=[1, 256, 64, 64], out.h=[1, 128, 64, 64]  

        # 15-th, h=[1, 128, 64, 64], cat_in=[1, 256, 64, 64], out.h=[1, 128, 64, 64]  

        import ipdb; ipdb.set_trace()
        return self.out(h) # h=[1, 128, 64, 64] to -> [1, 6, 64, 64] for what? NOTE
        # 6, first 3 for mean and next 3 for variance (all prediction)
    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        import ipdb; ipdb.set_trace() # TODO not used?
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        import ipdb; ipdb.set_trace()
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)

