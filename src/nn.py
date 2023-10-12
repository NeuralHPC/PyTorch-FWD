"""
Various utilities for neural networks as in https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/nn.py.
"""
import argparse
import math
from functools import partial
from typing import List

import torch
import torch.nn as nn

from src.freq_math import forward_wavelet_packet_transform


class SiLU(nn.Module):
    def forward(self, x):
        return nn.functional.silu(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def get_loss_fn(args: argparse.NameSpace, weights=None):
    if args.loss_type.lower() == "mse":
        return nn.MSELoss()
    elif args.loss_type.lower() == "packet":
        return partial(
            PacketLoss,
            wavelet=args.wavelet,
            level=args.max_level,
            norm_fn=args.packet_norm_type,
            norm_weights=weights,
        )
    elif args.loss_type.lower() == "mixed":
        return partial(
            MixedLoss,
            wavelet=args.wavelet,
            level=args.max_level,
            norm_fn=args.packet_norm_type,
            sigma=args.loss_sigma,
            norm_weight=weights,
        )


def PacketLoss(
    output: torch.Tensor,
    target: torch.Tensor,
    wavelet: str,
    level: int,
    norm_fn: str = None,
    norm_weights: List[float] = None,
) -> torch.Tensor:
    """Computes packet loss with various normalization techniques.

    Args:
        output (torch.Tensor): Network predictions
        target (torch.Tensor): Target values
        wavelet (str): Input wavelet
        level (int): Level of wavelet decomposition
        norm_fn (str, optional): norm function to use. Defaults to None.
        norm_weights (List[float], optional): norm weights to use. Defaults to None.

    Returns:
        torch.Tensor: packet loss value
    """

    def weighted_packets(
        packets: torch.Tensor, norm_weights: List[float]
    ) -> torch.Tensor:
        """Weigh the packets based on wavelet norm.

        Args:
            packets (torch.Tensor): Input packets of shape (bs, packets, c, h, w)
            norm_weights (List[float]): norm weights to use

        Raises:
            ValueError: if weights are not provided

        Returns:
            torch.Tensor: Weighted packets
        """
        if norm_weights is None:
            raise ValueError(
                "Norm weights must be provided as a list, else use log scale norm."
            )
        norm_weights = torch.reshape(torch.Tensor(norm_weights), (1, -1, 1, 1, 1))
        return norm_weights * packets

    def log_scale_packets(packets: torch.Tensor) -> torch.Tensor:
        """Normalize packets by log scaling.

        Args:
            packets (torch.Tensor): Packets of shape (bs, packets, c, h, w)

        Returns:
            torch.Tensor: log scale normalized packets
        """
        return torch.sign(packets) * torch.log(torch.abs(packets) + 1e-8)

    output_packets = forward_wavelet_packet_transform(output, wavelet, level)
    target_packets = forward_wavelet_packet_transform(target, wavelet, level)
    if norm_fn is not None:
        packet_norm = (
            log_scale_packets
            if "log" in norm_fn
            else partial(weighted_packets, norm_weights=norm_weights)
        )
        output_packets = packet_norm(output_packets)
        target_packets = packet_norm(target_packets)
    packet_mse = torch.mean(0.5 * (output_packets - target_packets) ** 2)
    return packet_mse


def MixedLoss(
    output: torch.Tensor,
    target: torch.Tensor,
    sigma: float,
    wavelet: str,
    level: int,
    norm_fn: str = None,
    norm_weights: List[float] = None,
) -> torch.Tensor:
    """Compute MSE with Packet loss.

    Args:
        output (torch.Tensor): Network predictions
        target (torch.Tensor): Target values
        sigma (float): weighting factor
        wavelet (str): Wavelet to use for packet loss
        level (int): Level of decomposition of wavelet tree
        norm_fn (str, optional): Normalization function for packet loss. Defaults to None.
        norm_weights (List[float], optional): normalized weights for packet loss. Defaults to None.

    Returns:
        torch.Tensor: Combination loss of MSE and Packet
    """
    mse_loss = nn.functional.mse_loss(output, target)
    packet_loss = PacketLoss(output, target, wavelet, level, norm_fn, norm_weights)
    return mse_loss + (sigma * packet_loss)
