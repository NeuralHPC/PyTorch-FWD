"""Sample the data in distributed manner."""

import torch
import torch.distributed as dist
from tqdm import tqdm
import os
from .sample import sample_DDPM
import math

def sample_ddp(model, sample_args)
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available(), "DDP Sampling needs atleast one GPU"
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = sample_args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    model.eval()

    # TODO: Create a sampling folder.
    dist.barrier()

    n = sample_args.per_proc_batch_size
    global_bs = n * dist.get_world_size()
    total_samples = int(math.ceil(sample_args.num_samples / global_bs) * global_bs)
    if rank == 0:
        print(f"{total_samples} number of images are sampled.")

    assert total_samples%dist.get_world_size() == 0, "Total number of samples must be divisible by world size"
    samples_this_device = int(total_samples / dist.get_world_size())
    assert samples_this_device%n == 0, "Samples for this device must be divisble by batch size"

    itrs = samples_this_device//n
    progress = tqdm(range(itrs)) if rank == 0 else range(itrs)
    total_imgs_sampled = 0
    for _ in progress:
        raise NotImplementedError
