from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from src.freq_math import (
    forward_wavelet_packet_transform,
    inverse_wavelet_packet_transform,
)


@jax.jit
def pad_odd(input_x: jnp.ndarray) -> jnp.ndarray:
    # dont pad the batch axis.
    pad_list = [(0, 0, 0)]
    for axis_shape in input_x.shape[1:-1]:
        if axis_shape % 2 != 0:
            pad_list.append((0, 1, 0))
        else:
            pad_list.append((0, 0, 0))
    # dont pad the features
    pad_list.append((0, 0, 0))
    return jax.lax.pad(input_x, 0.0, pad_list)


class UNet(nn.Module):
    output_channels: int
    transpose_conv = False
    wavelet_packets: int = False
    base_feat_no = 64  # TODO!

    @nn.compact
    def __call__(self, x_in: Tuple[jnp.ndarray]):
        # Downscaling block
        def down_block(x_bin, feats, time_lbl):
            y = nn.relu(
                nn.Conv(features=feats, kernel_size=(3, 3), padding="SAME")(x_bin)
            )
            time, label = time_lbl
            time_emb = nn.Dense(np.prod(y.shape[1:]))(time)
            lbl_emb = nn.Dense(np.prod(y.shape[1:]))(label)
            time_lbl_emb = time_emb + lbl_emb
            time_lbl_emb = jnp.reshape(time_lbl_emb, y.shape)
            y = y + time_lbl_emb
            y = nn.relu(
                nn.Conv(
                    features=feats, kernel_size=(3, 3), strides=(2, 2), padding="SAME"
                )(y)
            )
            y = nn.GroupNorm()(y)
            return pad_odd(y)

        # Upscaling block
        def up_block(x_bin, x_cat, feats, time_lbl):
            B, H, W, C = x_bin.shape
            if self.transpose_conv:
                y = nn.ConvTranspose(
                    features=feats, kernel_size=(3, 3), strides=(2, 2)
                )(x_bin)
            else:
                y = jax.image.resize(x_bin, (B, H * 2, W * 2, C), "nearest")
            y = y[:, : x_cat.shape[1], : x_cat.shape[2], :]
            # y_cat = jnp.concatenate([x_cat, y], axis=-1)
            y = nn.relu(nn.Conv(features=feats, kernel_size=(3, 3), padding="SAME")(y))
            time, label = time_lbl
            time_emb = nn.Dense(np.prod(y.shape[1:]))(time)
            lbl_emb = nn.Dense(np.prod(y.shape[1:]))(label)
            time_lbl_emb = time_emb + lbl_emb
            time_lbl_emb = jnp.reshape(time_lbl_emb, y.shape)
            y = y + time_lbl_emb
            y = nn.relu(nn.Conv(features=feats, kernel_size=(3, 3), padding="SAME")(y))
            y = nn.GroupNorm()(y)
            return y

        x, time, label = x_in
        # x_in = jnp.expand_dims(x, -1)
        x_in = x
        init_feat = self.base_feat_no
        time_lbl = time, label

        # Encoder
        x1 = nn.relu(
            nn.Conv(features=init_feat, kernel_size=(3, 3), padding="SAME")(x_in)
        )
        x2 = down_block(x1, init_feat, time_lbl)
        x3 = down_block(x2, init_feat * 2, time_lbl)
        x4 = down_block(x3, init_feat * 4, time_lbl)
        x5 = down_block(x4, init_feat * 8, time_lbl)
        x6 = nn.relu(
            nn.Conv(features=init_feat * 8, kernel_size=(3, 3), padding="SAME")(x5)
        )
        x7 = down_block(x6, init_feat * 16, time_lbl)
        x8 = x7 + nn.relu(
            nn.Conv(features=init_feat * 16, kernel_size=(3, 3), padding="SAME")(x7)
        )
        # Decoder
        x9 = up_block(x8, x6, init_feat * 8, time_lbl)
        x9 = x9 + x6
        x10 = up_block(x9, x5, init_feat * 8, time_lbl)
        x10 = x10 + x5
        x11 = up_block(x10, x4, init_feat * 4, time_lbl)
        x11 = x11 + x4
        x12 = up_block(x11, x3, init_feat * 2, time_lbl)
        x12 = x12 + x3
        x13 = up_block(x12, x2, init_feat, time_lbl)
        x13 = x13 + x2
        x14 = up_block(x13, x1, init_feat, time_lbl)
        y = nn.Conv(features=self.output_channels, kernel_size=(1, 1), padding="SAME")(
            x14
        )
        return y


# class Packetformer(nn.module):
#    pass
