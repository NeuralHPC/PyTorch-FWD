"""Torch DDP training script for diffusion."""
import argparse
import os.path
from typing import Tuple, Union, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

from src.freq_math import fourier_power_divergence, wavelet_packet_power_divergence
from src.sample_util import linear_noise_scheduler, sample_noise, sample_DDPM
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
        std: List[float] = [],
        mean: List[float] = [],
        input_shape: int = 32
    ) -> None:
        """DDP Train class.

        Args:
            model (nn.Module): Model Instance
            args (argparse.Namespace): User defined arguments
            optimizer (optim.Optimizer): Optimizer
            loss_fn (nn.Module): Loss function
            writer (Union[SummaryWriter, None]): tensorboard writer object for process 0 else None
            save_path (str): Path to save the model
            std (List[float]): Standard deviation of the dataset if any. Defaults to empty list
            mean (List[float]): Mean of the dataset if any. Defaults to empty list
            input_shape (int): Input dimension shape
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
        self.__std, self.__mean = torch.empty((1, 3, 1, 1)), torch.empty((1, 3, 1, 1))
        if len(std) != 0 and len(mean) != 0:
            self.__std = torch.reshape(torch.tensor(std), (1, 3, 1, 1)).to(
                self.__local_rank
            )
            self.__mean = torch.reshape(torch.tensor(mean), (1, 3, 1, 1)).to(
                self.__local_rank
            )
        self.__input_dim = input_shape

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
                last_epoch = epoch == max_epochs - 1
                if (epoch % 5 == 0) or last_epoch:
                    # Sample validation set
                    sample_imgs, original_imgs = self.__validation_sample(32, dataloader)
                    self.__tensorboard.add_images("Original images", original_imgs[:8, :, :, :], epoch)
                    self.__tensorboard.add_images("Sampled images", sample_imgs[:8, :, :, :], epoch)
                    # Compute fourier and wavelet power spectrum loss
                    fft_ab, fft_ba = fourier_power_divergence(sample_imgs, original_imgs)
                    packet_ab, packet_ba = wavelet_packet_power_divergence(sample_imgs, original_imgs)
                    fft_mean = 0.5 * (fft_ab + fft_ba)
                    packet_mean = 0.5 * (packet_ab + packet_ba)
                    self.__tensorboard.add_scalar("PS_fft_KLD", fft_mean.item(), epoch)
                    self.__tensorboard.add_scalar("PS_packet_KLD", packet_mean.item(), epoch)
                    self.model.train()
                if (epoch % self.__save_every == 0) or last_epoch:
                    self.__save_checkpoint(epoch)

    def __validation_sample(self, bs: int, dataloader: DataLoader) -> Tuple[torch.Tensor]:
        """Generate samples for validation purpose.

        Args:
            bs (int): Batch size for validation
            dataloader (DataLoader): Training datalaoder for labels
        
        Returns:
            Tuple[torch.Tensor]: Tuple containing sampled and original images
        """
        with torch.no_grad():
            self.model.eval()
            imgs, labels = next(iter(dataloader))
            imgs, labels = imgs[:bs, :, :, :], labels[:bs]
            imgs = imgs.to(self.__local_rank)

            x_0 = sample_DDPM(
                class_labels=labels,
                model=self.model,
                max_steps=self.__time_steps,
                input_shape=(bs, 3, self.__input_dim, self.__input_dim),
                device=self.__local_rank
            )

            if self.__std.nelement() != 0  and self.__mean.nelement() != 0:
                x_0 = (x_0 * self.__std) + self.__mean
                imgs = (imgs * self.__std) + self.__mean
            assert x_0.shape == imgs.shape, "generated images must be same shape as input."
            return x_0, imgs

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
