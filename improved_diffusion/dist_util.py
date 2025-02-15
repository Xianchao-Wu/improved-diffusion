"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group. 分布式训练处理group，不必看
    """
    if dist.is_initialized():
        return

    comm = MPI.COMM_WORLD # <mpi4py.MPI.Intracomm object at 0x7f78fb0e1570>
    backend = "gloo" if not th.cuda.is_available() else "nccl" # 'nccl'

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn()) # '172.17.0.3'
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0) # '172.17.0.3'
    os.environ["RANK"] = str(comm.rank) # '0'
    os.environ["WORLD_SIZE"] = str(comm.size) # '1'

    port = comm.bcast(_find_free_port(), root=0) # 45133
    os.environ["MASTER_PORT"] = str(port) # '45133'
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed. GPU`设备
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0) # NOTE 这是从一个gpu传递到其他的gpu了


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
