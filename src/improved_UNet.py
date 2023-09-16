"""UNet as implemented in https://arxiv.org/abs/2102.09672."""

import math
from abc import abstractmethod
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    SiLU,
    avg_pool_nd,
    checkpoint,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)


class TimeStepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        """Forward pass for the embedding block.

        Args:
            x (torch.Tensor): Input tensor
            emb (torch.Tensor): Time and class embedding
        """


class TimeStepEmbedSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with embedding.

        Args:
            x (torch.Tensor): Input tensor
            emb (torch.Tensor): Time and class embedding

        Returns:
            torch.Tensor: Output tensor
        """
        for layer in self:
            if isinstance(layer, TimeStepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class UpSample(nn.Module):
    """Upsampling layer with an optional convolution."""

    def __init__(self, channels: int, use_conv: bool, dims: int = 2) -> None:
        """Initialize upsample layer.

        Args:
            channels (int): Output channels
            use_conv (bool): Boolean specifying to use convolution.
            dims (int, optional): Dimensions for convolution. Defaults to 2.
        """
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape [N, C, spatial_dims]

        Returns:
            torch.Tensor: Upsampled tesnor of shape [N, C, 2*spatial_dims]
        """
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class DownSample(nn.Module):
    """Downsampling layer"""

    def __init__(self, channels: int, use_conv: bool, dims: int = 2):
        """Initialize downsample layer.

        Args:
            channels (int): Output channels
            use_conv (bool): Boolean specifying to use convolution
            dims (int, optional): Dimensions for convolution. Defaults to 2.
        """
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.multiplication = conv_nd(
                dims, channels, channels, 3, stride=stride, padding=1
            )
        else:
            self.multiplication = avg_pool_nd(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape [N, C, spatial_dims]

        Returns:
            torch.Tensor: Downsampled tesnor of shape [N, C, spatial_dims/2]
        """
        assert x.shape[1] == self.channels
        return self.multiplication(x)


class ResBlock(TimeStepBlock):
    """Residual Block with dynamic channels."""

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Union(int, None) = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
    ) -> None:
        """Residual block for each UNet step.

        Args:
            channels (int): Input channels
            emb_channels (int): Embedded channels
            dropout (float): Dropout probability
            out_channels (Union, optional): Output channels. Defaults to None.
            use_conv (bool, optional): Boolean for using convolution. Defaults to False.
            use_scale_shift_norm (bool, optional): Boolean for scale shift norm. Defaults to False.
            dims (int, optional): Dimensions for convolution. Defaults to 2.
            use_checkpoint (bool, optional): Use checkpoint. Defaults to False.
        """
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.droput = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return checkpoint(self.call, (x, emb), self.parameters(), self.use_checkpoint)

    def call(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class Attention(nn.Module):
    """Attention block."""

    def __init__(
        self, channels: int, num_heads: int = 1, use_checkpoint: bool = False
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self.call, (x,), self.parameters(), self.use_checkpoint)

    def call(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.scaled_dot_product(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

    def scaled_dot_product(self, x: torch.Tensor) -> torch.Tensor:
        ch = x.shape[1] // 3
        q, k, v = torch.split(x, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct, bcs -> bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts, bcs -> bct", weight, v)


class Improv_UNet(nn.Module):
    """UNet model with attention and timestep embedding."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Tuple[int],
        dropout: float = 0,
        channel_mult: Tuple[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Union(int, None) = None,
        use_checkpoint: bool = False,
        num_heads: int = 1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
    ) -> None:
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimeStepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1),
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        Attention(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimeStepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimeStepEmbedSequential(DownSample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_blocks = TimeStepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            Attention(
                ch,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        Attention(
                            ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(UpSample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimeStepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, y: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass."""
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_blocks(h, emb)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)


class SuperResModel(Improv_UNet):
    """UNet Model to perform super resolution."""

    def __init__(self, in_channels: int, *args, **kwargs) -> None:
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        low_res: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        _, _, new_height, new_width = x.shape
        upsampled_res = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = torch.cat((x, upsampled_res), dim=1)
        return super().forward(x, timesteps, **kwargs)
