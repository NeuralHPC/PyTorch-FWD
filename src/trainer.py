"""Torch DDP training script for diffusion."""
import os

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from time import time

from .dataloader import get_distributed_dataloader, load_input
from .nn import get_loss_fn


def clean_ddp():
    """
    Clear process group.
    """
    dist.destroy_process_group()


def logger():
    raise NotImplementedError


def train(model, datasets, log_direc, train_args):
    """Training model DDP."""
    assert torch.cuda.is_available(), "Need atleast one GPU for training.."

    dist.init_process_group("nccl")  # NCCL backend for distributed training.
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert train_args.batch_size % world_size == 0, "Batch size and world size must be perfectly divisible.."

    device = rank % torch.cuda.device_count()
    local_seed = train_args.global_seed * world_size + rank
    torch.manual_seed(local_seed)
    torch.cuda.set_device(device)

    # TODO: Setup logging here.

    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=train_args.lr)
    criterion = get_loss_fn(train_args.loss)

    if isinstance(datasets, tuple):
        train_set, _ = datasets
    else:
        train_set = datasets
    train_loader, train_sampler = get_distributed_dataloader(train_set,
                                                             world_size,
                                                             rank,
                                                             train_args.global_seed,
                                                             train_args.batch_size,
                                                             train_args.num_workers)

    model.train()
    time_steps = train_args.time_steps
    train_steps = 0
    running_loss = 0.
    log_steps = 0
    start_time = time()
    for epoch in range(train_args.epochs):
        train_sampler.set_epoch(epoch)
        for i, (input_img, class_label) in enumerate(train_loader):
            x, y, current_step = load_input(input_img, time_steps)
            x, y, class_label = x.to(device), y.to(device), class_label.to(device)
            current_steps = current_steps.to(device)
            pred_noise = model(x, current_steps, class_label)
            loss_val = criterion(pred_noise, y)
            optimizer.zero_grad()
            loss_val.backward()
            # Perform gradient clipping
            if train_args.clip_value > 0.0:
                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_args.clip_value)
                except Exception:
                    pass

            optimizer.step()

            running_loss += loss_val.item()
            log_steps += 1
            train_steps += 1

            if train_steps % 100 == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                print(f"Train step: {train_steps:.07d},\
                        Train Loss: {avg_loss:.4f},\
                        Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0.
                log_steps = 0
                start_time = time()

        # Save every 100th epoch or last training epoch
        if (epoch % 100 == 0 and epoch > 0) or (epoch == train_args.epochs - 1):
            if rank == 0:
                check_point = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_args": train_args
                }
                ckpt_path = f"{log_direc}/checkpoints/"
                os.makedirs(ckpt_path, exist_ok=True)
                ckpt_path += f"{epoch}.pt"
                torch.save(check_point, ckpt_path)
            dist.barrier()
    clean_ddp()
