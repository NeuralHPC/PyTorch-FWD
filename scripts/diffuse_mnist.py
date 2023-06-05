import argparse
from typing import List
from functools import partial

import jax
import jax.numpy as jnp

import optax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

import numpy as np
import matplotlib.pyplot as plt


from datasets import load_dataset
from src.util import _parse_args, pad_odd


class UNet(nn.Module):

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x1 = jnp.expand_dims(x, -1)
        init_feat = 16
        # out_neurons = 2

        def down_block(x_in, feats):
            y = nn.relu(nn.Conv(
                features=feats, kernel_size=(3, 3), padding="SAME")(x_in))
            y = nn.relu(nn.Conv(features=feats,
                                kernel_size=(3, 3), padding="SAME")(y))
            y = pad_odd(y)
            return nn.max_pool(y, (3, 3), strides=(2, 2), padding='SAME')

        x2 = down_block(x1, init_feat)
        x3 = down_block(x2, init_feat * 2)
        x4 = down_block(x3, init_feat * 4)
        x5 = down_block(x4, init_feat * 8)

        # breakpoint()
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
        x10 = up_block(x9, x1, init_feat)  # TODO: Too small??
        y = nn.Conv(
            features=1, kernel_size=(3, 3), padding="SAME")(x10)
        return y


@partial(jax.jit, static_argnames=['model'])
def diff_step(net_state, noisy_batch, batch, model):
    denoise = model.apply(net_state, noisy_batch)
    cost = jnp.mean(0.5 * (jnp.expand_dims(batch, -1) - denoise) ** 2)
    return cost


loss_grad_fn = jax.value_and_grad(diff_step, argnums=0)


@partial(jax.jit, static_argnames=['model', 'opt', 'time_steps'])
def train_step(batch: jnp.ndarray,
               net_state: FrozenDict, model: nn.Module,
               opt_state: FrozenDict, opt: optax.GradientTransformation,
               key: jax.random.PRNGKey, time_steps: int):

    key = jax.random.split(key, 1)[0]
    noise_array = jax.random.uniform(
        key, [time_steps] + list(batch.shape),
        minval=-.8, maxval=.8)
    cum_noise_array = jnp.cumsum(noise_array, axis=0)

    x_array = jnp.expand_dims(batch, 0) + cum_noise_array
    y_array = jnp.expand_dims(batch, 0) + jnp.concatenate(
        [jnp.zeros([1] + list(batch.shape)), cum_noise_array[:-1]])
    map_array = jnp.stack([x_array, y_array], 1)

    def reconstruct(map_array, net_state, model):
        x = map_array[:, 0]
        y = map_array[:, 1]
        cel, grads = loss_grad_fn(net_state, x, y, model)
        return cel, grads

    time_map = jax.vmap(
        partial(reconstruct, net_state=net_state, model=model))
    cels, grads = time_map(map_array)
    grads = jax.tree_map(partial(jnp.mean, axis=0), grads)
    cels = jnp.mean(cels)
    updates, opt_state = opt.update(grads, opt_state, net_state)
    net_state = optax.apply_updates(net_state, updates)
    return cels, net_state, opt_state


def test(net_state: FrozenDict, model: nn.Module, key: int,
         input_shape: List[int], time_steps: int):
    key = jax.random.PRNGKey(time_steps)
    process_array = jax.random.uniform(
        key, [1] + input_shape,
        minval=-.8, maxval=.8)
    for _ in range(time_steps):
        process_array = model.apply(net_state, process_array)[:, :, :, 0]
    return process_array


if __name__ == '__main__':
    args = _parse_args()

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    dataset = load_dataset("mnist")
    train_data = np.stack([np.array(img) for img in dataset['train']['image']])
    train_batches = np.array_split(
        train_data,
        len(dataset['train']['image']) // args.batch_size)

    input_shape = list(np.array(dataset['train']['image'][0]).shape)
    stats = {"mean": np.mean(train_data),
             "std": np.std(train_data)}

    model = UNet()
    opt = optax.adam(0.001)
    # create the model state
    net_state = model.init(key, jnp.ones([args.batch_size] + input_shape))
    opt_state = opt.init(net_state)

    for e in range(args.epochs):
        for pos, img in enumerate(train_batches):
            img = jnp.array(img)
            img_norm = (img - stats["mean"]) / stats["std"]
            mean_loss, net_state, opt_state = train_step(
                img_norm, net_state, model,
                opt_state, opt,
                key, args.time_steps)
            print(e, pos, mean_loss, len(train_batches))

    breakpoint()
    # testing...
    test_image = test(net_state, model, 4, input_shape, 30)
    breakpoint()
