import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model, # <class 'improved_diffusion.unet.UNetModel'>
        diffusion, # <class 'improved_diffusion.respace.SpacedDiffusion'>
        data, # <class 'generator'>
        batch_size, # 128
        microbatch, # -1
        lr, # 0.0001
        ema_rate, # '0.9999'
        log_interval, # 10
        save_interval, # 10000
        resume_checkpoint, # ''
        use_fp16=False,
        fp16_scale_growth=1e-3, # 0.001
        schedule_sampler=None, # <improved_diffusion.resample.LossSecondMomentResampler object at 0x7f4db0788490>
        weight_decay=0.0, # 0.0
        lr_anneal_steps=0, # 0
    ):
        import ipdb; ipdb.set_trace()
        self.model = model # UNetModel
        self.diffusion = diffusion # SpacedDiffusion
        self.data = data # generator, dataset, <generator object load_data at 0x7f6ad86b7740>
        self.batch_size = batch_size # 128
        self.microbatch = microbatch if microbatch > 0 else batch_size # 128
        self.lr = lr # 0.0001
        self.ema_rate = (
            [ema_rate] # '0.9999'
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        ) # [0.9999]
        self.log_interval = log_interval # 10
        self.save_interval = save_interval # 10000
        self.resume_checkpoint = resume_checkpoint # ''
        self.use_fp16 = use_fp16 # False
        self.fp16_scale_growth = fp16_scale_growth # 0.001
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion) # 这个是使用已有的LossSecondMomentResampler NOTE
        self.weight_decay = weight_decay # 0.0
        self.lr_anneal_steps = lr_anneal_steps # 0

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size() # 128*1 = 128

        self.model_params = list(self.model.parameters()) # 451个元素
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE # 20.0
        self.sync_cuda = th.cuda.is_available() # True
        import ipdb; ipdb.set_trace() # NOTE
        self._load_and_sync_parameters()
        if self.use_fp16: # False
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay) # len(self.master_params)=451, self.lr=0.0001, 0.0
        if self.resume_step: # not in
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate)) # NOTE here
            ] # len(self.ema_params)=1, len(self.ema_params[0])=451

        import ipdb; ipdb.set_trace()

        if th.cuda.is_available():
            self.use_ddp = True # NOTE here now...
            # for debug, do not use DDP
            debug = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()], # device(type='cuda', index=0)
                output_device=dist_util.dev(), # device(type='cuda', index=0)
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            ) if not debug else self.model
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint # '', not resume checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
        import ipdb; ipdb.set_trace() # NOTE this is important! 在多个gpu之间同步参数
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        import ipdb; ipdb.set_trace()
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data) # batch.shape=[1, 3, 64, 64]; cond={'y': tensor([1])}
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            ''' e.g., 
            ------------------------
            | grad_norm | 37       |
            | loss      | 23.4     |
            | loss_q0   | 83.8     |
            | loss_q2   | 5.59     |
            | loss_q3   | 19.8     |
            | samples   | 6        |
            | step      | 0        |
            ------------------------

            '''
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
        # NOTE saved checkpoint space=
        '''root@95f4c42cafe7:/tmp# find ./* -name *.pt
        ./openai-2023-01-15-23-52-42-091905/ema_0.9999_000000.pt
        ./openai-2023-01-15-23-52-42-091905/model000000.pt
        ./openai-2023-01-15-23-52-42-091905/opt000000.pt
        '''

    def run_step(self, batch, cond): # batch.shape=[1, 3, 64, 64]; cond={'y': tensor([1])
        import ipdb; ipdb.set_trace()
        self.forward_backward(batch, cond)
        if self.use_fp16: # False, not in
            self.optimize_fp16()
        else:
            self.optimize_normal() # NOTE here, in
        self.log_step()

    def forward_backward(self, batch, cond): # batch.shape=[1,3,64,64]; cond={'y': tensor([1])
        import ipdb; ipdb.set_trace()
        zero_grad(self.model_params) # len(self.model_params)=451, 
        for i in range(0, batch.shape[0], self.microbatch): # batch.shape[0]=1, self.microbatch=1
            micro = batch[i : i + self.microbatch].to(dist_util.dev()) # [1,3,64,64]
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            } # {'y': tensor([1], device='cuda:0')}
            last_batch = (i + self.microbatch) >= batch.shape[0] # True, last_micro_batch (in current batch)
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev()) # NOTE, 这是采样出来一个t=[2801, 3376, ...] 取值是0到4000，以及weights=[1., 1., ...]，目前这个都是1了. dist_util.dev()="device(type='cuda', index=0)"
            import ipdb; ipdb.set_trace()
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro, # [1, 3, 64, 64], a micro batch
                t, # e.g., tensor([2801], device='cuda:0')
                model_kwargs=micro_cond, # {'y': tensor([1], device='cuda:0')}
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses() # NOTE, here
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses( # NOTE in
                    t, losses["loss"].detach()
                )
            import ipdb; ipdb.set_trace()
            loss = (losses["loss"] * weights).mean() # NOTE 这个weights都有哪些可能的取值呢? 目前都是1., ..., 1.
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16: # False, not in
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        import ipdb; ipdb.set_trace()
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params): # e.g., self.ema_rate=0.9999
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
