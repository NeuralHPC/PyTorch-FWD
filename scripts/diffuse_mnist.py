import argparse

import jax
import jax.numpy as jnp

import optax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

import numpy as np

from datasets import load_dataset
from src.util import _parse_args, pad_odd


class UNet(nn.Module):

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x1 = jnp.expand_dims(x, -1)
        init_feat = 8  # 16
        # out_neurons = 2

        def down_block(x_in, feats):
            y = nn.relu(nn.Conv(
                features=feats, kernel_size=(3, 3), padding="SAME")(x_in))
            y = nn.relu(nn.Conv(features=feats,
                                kernel_size=(3, 3), padding="SAME")(y))
            y = pad_odd(x1)
            return nn.max_pool(y, (2, 2), strides=(2, 2))

        x2 = down_block(x1, init_feat)
        x3 = down_block(x2, init_feat * 2)
        x4 = down_block(x3, init_feat * 4)
        x5 = down_block(x4, init_feat * 8)

        x6 = nn.relu(nn.Conv(
            features=init_feat * 16, kernel_size=(3, 3), padding="SAME")(x5))
        x6 = nn.relu(nn.Conv(
            features=init_feat * 16, kernel_size=(3, 3), padding="SAME")(x6))

        def up_block(x_in, x_cat, feats):
            y = nn.ConvTranspose(
                features=feats, kernel_size=(3, 3), strides=(2, 2))(x_in)
            y = y[:, :x_cat.shape[1], :x_cat.shape[2], :]
            y_cat = jnp.concatenate([x_cat, y], axis=-1)
            y = nn.relu(nn.Conv(
                features=feats, kernel_size=(3, 3), padding="SAME")(y_cat))
            y = nn.relu(nn.Conv(
                features=feats, kernel_size=(3, 3), padding="SAME")(y))
            return y

        x7 = up_block(x6, x4, init_feat * 8)
        x8 = up_block(x7, x3, init_feat * 4)
        x9 = up_block(x8, x2, init_feat * 2)
        x10 = up_block(x9, x1, 3)  # TODO: Too small??
        return x10


def train_step(batch: jnp.ndarray, net_state: FrozenDict,
               opt_state: FrozenDict, model: nn.Module,
               key: jax.random.PRNGKey, args: argparse.Namespace):
    breakpoint()
    for t in range(args.time_steps):
        key = jax.random.split(key, 1)
        noisy_batch = batch + jax.random.uniform(key, batch.shape)
        pass


if __name__ == '__main__':
    args = _parse_args()

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    dataset = load_dataset("mnist")
    train_batches = np.array_split(
        [np.array(img) for img in dataset['train']['image']],
        len(dataset['train']['image']) // args.batch_size)

    input_shape = list(np.array(dataset['train']['image'][0]).shape)

    model = UNet()
    opt = optax.adam(0.001)
    # create the model state
    net_state = model.init(key, jnp.ones([args.batch_size] + input_shape))
    opt_state = opt.init(net_state)

    for img in train_batches:
        img = jnp.array(img)
        train_step(img, net_state, opt_state, model, key, args)
    pass
