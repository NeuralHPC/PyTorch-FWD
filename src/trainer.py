"""Torch DDP training script for diffusion."""
import argparse
import os.path
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

from src.sample_util import linear_noise_scheduler, sample_noise
from src.util import _get_global_rank, _get_local_rank


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        args: argparse.Namespace,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        writer: Union[SummaryWriter, None],
        save_path: str,
    ) -> None:
        """DDP Train class.

        Args:
            model (nn.Module): Model Instance
            args (argparse.Namespace): User defined arguments
            optimizer (optim.Optimizer): Optimizer
            loss_fn (nn.Module): Loss function
            writer (Union[SummaryWriter, None]): tensorboard writer object for process 0 else None
            save_path (str): Path to save the model
        """
        # Initialize ranks of device.
        self.__local_rank = _get_local_rank()
        self.__global_rank = self.__local_rank
        if args.distribute:
            self.__global_rank = _get_global_rank()

        self.__elapsed_epochs = 0
        # Initialize the model.
        self.model = model.to(self.__local_rank)
        if args.model_path is not None:
            if os.path.exists(args.model_path):
                if self.__global_rank == 0:
                    self.__load_checkpoint(args.model_path)
        self.model = DDP(
            self.model, device_ids=[self.__local_rank], find_unused_parameters=True
        )

        # Initialize training variables
        self.__optimizer = optimizer
        self.__loss_fn = loss_fn
        self.__tensorboard = writer
        self.__clip_grad_norm = args.clip_grad_norm
        self.__time_steps = args.time_steps
        self.__save_every = args.save_every
        self.__save_path = save_path
        self.__print_every = args.print_every

    def __load_checkpoint(self, path: str) -> None:
        """Load the existing checkpoint.

        Args:
            path (str): Existing model path
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        self.__elapsed_epochs = checkpoint["elapsed_epochs"] + 1
        print(
            f"[GPU{self.__global_rank}] Resuming training from epoch: {self.__elapsed_epochs}"
        )

    def __save_checkpoint(self, epoch: int) -> None:
        """Save checkpoint.

        Args:
            epoch (int): Current epoch
        """
        checkpoint = {}
        checkpoint["model_state"] = self.model.module.state_dict()
        checkpoint["elapsed_epochs"] = epoch
        file_path = os.path.join(self.__save_path, f"model_ckpt_epoch_{epoch}.pt")
        torch.save(checkpoint, file_path)
        print(f"===> Checkpoint saved at epoch {epoch} to {file_path}")

    def __train_step(self, dataloader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            dataloader (DataLoader): Training dataloader

        Returns:
            float: Epoch loss
        """
        self.model.train()
        per_step_loss = 0.0
        total_steps = 0
        self.model.train()
        for input, class_label in dataloader:
            # Preprocess and load input to corresponding device.
            x, y, current_steps = self.__preprocess_input(input)
            x, y = x.to(self.__local_rank), y.to(self.__local_rank)
            class_label, current_steps = class_label.to(
                self.__local_rank
            ), current_steps.to(self.__local_rank)

            # Forward pass and loss calculation
            self.__optimizer.zero_grad()
            pred_noise = self.model(x, current_steps, class_label)
            loss_val = self.__loss_fn(pred_noise, y)

            # Backward pass
            loss_val.backward()
            if self.__clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.__clip_grad_norm
                )
            self.__optimizer.step()
            per_step_loss += loss_val.item()
            total_steps += 1
            if total_steps % self.__print_every == 0 and self.__global_rank == 0:
                print(f"Step: {total_steps}, Loss: {per_step_loss / total_steps}")
        avg_loss = per_step_loss / total_steps
        return avg_loss

    def train(self, max_epochs: int, dataloader: DataLoader, sampler: Sampler) -> None:
        """Training looooooop.

        Args:
            max_epochs (int): Total epochs
            dataloader (DataLoader): Training dataloader
            sampler (Sampler): Datasampler
        """
        for epoch in range(self.__elapsed_epochs, max_epochs):
            if self.__global_rank == 0:
                print(f"Epoch: {epoch}, total_batches: {len(dataloader)}")

            sampler.set_epoch(epoch)
            epoch_loss = self.__train_step(dataloader)

            if self.__global_rank == 0:
                print(f"Training loss: {epoch_loss}", flush=True)
                self.__tensorboard.add_scalar("Train Loss", epoch_loss, epoch)
                self.__tensorboard.flush()
                # TODO: Perform generation of 10 images for evaluation purpose
                # TODO: Log fft PSKL and packet PSKL
                if (epoch % self.__save_every == 0) or (epoch == max_epochs - 1):
                    self.__save_checkpoint(epoch)

    def __preprocess_input(self, input_imgs: torch.Tensor) -> Tuple[torch.Tensor]:
        """Apply noising to the input images.

        Args:
            input_imgs (torch.Tensor): Input images of shape [BSxCxHxW]

        Returns:
            Tuple[torch.Tensor]: Tuple containing noised input, noise and step values.
        """
        current_steps = torch.randint(
            high=self.__time_steps, size=[input_imgs.shape[0]]
        )
        alphas_t = torch.tensor(
            [
                linear_noise_scheduler(time, self.__time_steps)[0]
                for time in current_steps
            ]
        ).reshape(len(current_steps), 1)
        batch_map = torch.vmap(sample_noise, randomness="different")
        x, y = batch_map(input_imgs, alphas_t)
        return x, y, current_steps
