"""Sample the data in distributed manner."""

import argparse
import os
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.sample_util import sample_DDIM, sample_DDPM
from src.util import _get_global_rank, _get_local_rank


class Sampler:
    def __init__(
        self,
        model: nn.Module,
        args: argparse.Namespace,
        save_path: str,
        std: List[float] = [],
        mean: List[float] = [],
    ) -> None:
        """DDP Sample class.

        Args:
            model (nn.Module): Model Instance
            args (argparse.Namespace): User defined arguments
            save_path (str): Sample save path
            std (List[float]): Standard deviation if any. Defaults to empty list.
            mean (List[float]): Mean of dataset if any. Defaults to empty list.
        """
        self.__local_rank = _get_local_rank()
        self.__global_rank = (
            _get_global_rank() if args.distribute else self.__local_rank
        )

        self.model = model.to(self.__local_rank)
        if os.path.exists(args.ckpt_path):
            if self.__global_rank == 0:
                self.__load_checkpoint(args.ckpt_path)
        else:
            raise ValueError(
                "Needs a trained model path. Please provide or check your path."
            )
        self.model = DDP(self.model, device_ids=[self.__local_rank])

        self.__save_path = save_path
        self.__input_shape = (args.batch_size, 3, args.input_shape, args.input_shape)
        self.__sampler = sample_DDPM if args.sampler == "DDPM" else sample_DDIM
        self.__std, self.__mean = torch.empty((1, 3, 1, 1)), torch.empty((1, 3, 1, 1))
        if len(std) != 0 and len(mean) != 0:
            self.__std = torch.reshape(torch.tensor(std), (1, 3, 1, 1)).to(
                self.__local_rank
            )
            self.__mean = torch.reshape(torch.tensor(mean), (1, 3, 1, 1)).to(
                self.__local_rank
            )
        self.__timesteps = args.diff_steps

    def __load_checkpoint(self, path: str) -> None:
        """Load the sampling checkpoint.

        Args:
            path (str): Model path
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        print("Model loaded successfully...")

    def conditional_sample(self, batched_labels: DataLoader) -> None:
        """Sample batch of images sampled conditionally.

        Args:
            batched_labels (DataLoader): Batches of class labels.
        """
        with torch.no_grad():
            self.model.eval()
            for _, labels in tqdm(batched_labels):
                x_0 = self.__sampler(
                    class_labels=labels,
                    model=self.model,
                    max_steps=self.__timesteps,
                    input_shape=self.__input_shape,
                    device=self.__local_rank,
                )

                if self.__std.nelement() != 0 and self.__mean.nelement() != 0:
                    x_0 = (x_0 * self.__std) + self.__mean

                np_imgs = x_0.cpu().permute(0, 2, 3, 1).numpy()
                str_time = str(time.time()).replace(".", "_")
                fname = f"batch_{str_time}.npz"
                fpath = os.path.join(self.__save_path, fname)
                np.savez(fpath, x=np_imgs)
                print(f"Saved batch at {fname}..", flush=True)
