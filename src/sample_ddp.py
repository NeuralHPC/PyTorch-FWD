"""Sample the data in distributed manner."""

import os

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class Sampler:
    def __init__(self, model: nn.Module, batch_size: int, img_size: int, sample_path: str):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank])
        self.noise_img_shape = (batch_size, 3, img_size, img_size)
        self.sample_path = sample_path
        pass

    def sample(self, total_batches):
        raise NotImplementedError
