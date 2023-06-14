from typing import Tuple

import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn


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
    return jax.lax.pad(input_x, 0., pad_list)


class UNet(nn.Module):
    transpose_conv = False
    base_feat_no = 64 # 128

    @nn.compact
    def __call__(self, x_in: Tuple[jnp.ndarray]):
        x, time = x_in
        x_in = jnp.expand_dims(x, -1)
        init_feat = self.base_feat_no


        x1 = nn.relu(nn.Conv(
                     features=init_feat, kernel_size=(3, 3), padding="SAME")(x_in))

        def down_block(x_bin, feats, time):
            y = nn.relu(nn.Conv(
                features=feats, kernel_size=(3, 3), padding="SAME")(x_bin))
            time_emb = nn.Dense(np.prod(y.shape[1:]))(time)
            time_emb = jnp.reshape(time_emb, [1] + list(y.shape[1:]))
            y = y + time_emb
            y = nn.relu(nn.Conv(features=feats,
                                kernel_size=(3, 3), strides=(2, 2),
                                padding="SAME")(y))
            y = nn.GroupNorm()(y)
            return pad_odd(y) 

        x2 = down_block(x1, init_feat, time)
        x3 = down_block(x2, init_feat * 2, time)
        x4 = down_block(x3, init_feat * 4, time)
        x5 = down_block(x4, init_feat * 8, time)

        x6 = x5 + nn.relu(nn.Conv(
            features=init_feat * 8, kernel_size=(3, 3), padding="SAME")(x5))

        def up_block(x_bin, x_cat, feats, time):
            B, H, W, C = x_bin.shape
            if self.transpose_conv:
                y = nn.ConvTranspose(
                    features=feats, kernel_size=(3, 3), strides=(2, 2))(x_bin)
            else:
                y = jax.image.resize(x_bin, (B, H * 2, W * 2, C), 'nearest')
            y = y[:, :x_cat.shape[1], :x_cat.shape[2], :]
            #y_cat = jnp.concatenate([x_cat, y], axis=-1)
            time_emb = nn.Dense(np.prod(y.shape[1:]))(time)
            time_emb = jnp.reshape(time_emb, [1] + list(y.shape[1:]))
            y = nn.relu(nn.Conv(
                features=feats, kernel_size=(3, 3), padding="SAME")(y))
            y = y + time_emb
            y = nn.relu(nn.Conv(
                features=feats, kernel_size=(3, 3), padding="SAME")(y))
            y = nn.GroupNorm()(y)
            return y

        x7 = up_block(x6, x4, init_feat * 4, time) # 4
        x7 = x7 + x4
        x8 = up_block(x7, x3, init_feat * 2, time) # 2
        x8 = x8 + x3
        x9 = up_block(x8, x2, init_feat, time)
        x9 = x9 + x2
        x10 = up_block(x9, x1, init_feat, time)  # TODO: Too small??
        y = nn.Conv(
            features=1, kernel_size=(1, 1), padding="SAME")(x10)
        return x_in + y