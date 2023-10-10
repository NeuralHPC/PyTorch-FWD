"""Sample the data in distributed manner."""

import os
import argparse

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from src.util import _get_global_rank, _get_local_rank
from src.sample import sample_DDPM, sample_DDIM


class Sampler:
    def __init__(self,
                 model: nn.Module,
                 args: argparse.Namespace,
                 save_path: str
                 ) -> None:
        """DDP Sample class.

        Args:
            model (nn.Module): Model Instance
            args (argparse.Namespace): User defined arguments
            save_path (str): Sample save path
        """
        self.__local_rank = _get_local_rank()
        self.__global_rank = _get_global_rank() if args.distribute else self.__local_rank

        self.model = model.to(self.__local_rank)
        if os.path.exists(args.ckpt_path):
            if self.__global_rank == 0:
                self.__load_checkpoint(args.ckpt_path)
        else:
            raise ValueError("Needs a trained model path. Please provide or check your path.")
        self.model = DDP(self.model, device_ids=[self.__local_rank])

        self.__save_path = save_path
        self.__input_shape = (args.batch_size, 3, args.input_shape, args.input_shape)
        self.__sampler = sample_DDPM if args.sampler == 'DDPM' else sample_DDIM
    

    def __load_checkpoint(self, path: str) -> None:
        """Load the sampling checkpoint.

        Args:
            path (str): Model path
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        print("Model loaded successfully...")

    def sample(self) -> torch.Tensor:
        """Sample batch of images

        Returns:
            torch.Tensor: Sampled batch of images.
        """
        raise NotImplementedError("Should be implemented... Important")

    

