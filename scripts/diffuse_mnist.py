import argparse
from typing import List, Tuple
from functools import partial

import jax
import jax.numpy as jnp
# jax.config.update('jax_threefry_partitionable', True)

import optax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

import numpy as np
import matplotlib.pyplot as plt


from datasets import load_dataset
from src.util import _parse_args, pad_odd, get_mnist_train_data


class UNet(nn.Module):
    transpose_conv = False
    base_feat_no = 32

    @nn.compact
    def __call__(self, x_in: Tuple[jnp.ndarray]):
        x, time = x_in
        x1 = jnp.expand_dims(x, -1)
        init_feat = self.base_feat_no
        time_emb = nn.Dense(np.prod(x.shape[1:]))(time)
        time_emb = jnp.reshape(time_emb, [1] + list(x1.shape[1:]))

        x1 = nn.swish(nn.Conv(
                     features=init_feat, kernel_size=(3, 3), padding="SAME")(x1))
        x1 = x1 + time_emb

        def down_block(x_in, feats):
            y = nn.swish(nn.Conv(
                features=feats, kernel_size=(3, 3), padding="SAME")(x_in))
            y = nn.swish(nn.Conv(features=feats,
                                kernel_size=(3, 3), strides=(2, 2),
                                padding="SAME")(y))
            return pad_odd(y) 

        x2 = down_block(x1, init_feat)
        x3 = down_block(x2, init_feat * 2)
        x4 = down_block(x3, init_feat * 4)
        x5 = down_block(x4, init_feat * 8)

        x6 = x5 + nn.swish(nn.Conv(
            features=init_feat * 8, kernel_size=(3, 3), padding="SAME")(x5))

        def up_block(x_in, x_cat, feats):
            B, H, W, C = x_in.shape
            if self.transpose_conv:
                y = nn.ConvTranspose(
                    features=feats, kernel_size=(3, 3), strides=(2, 2))(x_in)
            else:
                y = jax.image.resize(x_in, (B, H * 2, W * 2, C), 'nearest')
            y = y[:, :x_cat.shape[1], :x_cat.shape[2], :]
            y_cat = jnp.concatenate([x_cat, y], axis=-1)
            y = nn.swish(nn.Conv(
                features=feats, kernel_size=(3, 3), padding="SAME")(y_cat))
            y = nn.swish(nn.Conv(
                features=feats, kernel_size=(3, 3), padding="SAME")(y))
            return y

        x7 = up_block(x6, x4, init_feat * 4) # 4
        x7 = x7 + x4
        x8 = up_block(x7, x3, init_feat * 2) # 2
        x8 = x8 + x3
        x9 = up_block(x8, x2, init_feat)
        x9 = x9 + x2
        x10 = up_block(x9, x1, init_feat)  # TODO: Too small??
        y = nn.Conv(
            features=1, kernel_size=(1, 1), padding="SAME")(x10)
        return x1 + y


@partial(jax.jit, static_argnames=['model'])
def diff_step(net_state, noisy_batch, batch, time, model):
    denoise = model.apply(net_state, (noisy_batch, time))
    cost = jnp.mean(0.5 * (jnp.expand_dims(batch, -1) - denoise) ** 2)
    return cost


loss_grad_fn = jax.value_and_grad(diff_step, argnums=0)


@partial(jax.jit, static_argnames=['model', 'opt', 'time_steps'])
def train_step(batch: jnp.ndarray,
               net_state: FrozenDict, opt_state: FrozenDict,
               seed: int, model: nn.Module,
               opt: optax.GradientTransformation,
               time_steps: int):
    batch = batch
    key = jax.random.PRNGKey(seed[0])
    noise_array = jax.random.uniform(
        key, [time_steps] + list(batch.shape),
        minval=-1, maxval=1)
    cum_noise_array = jnp.cumsum(noise_array, axis=0)

    x_array = jnp.expand_dims(batch, 0) + cum_noise_array
    y_array = jnp.expand_dims(batch, 0) + jnp.concatenate(
        [jnp.zeros([1] + list(batch.shape)), cum_noise_array[:-1]])
    map_tree = (x_array, y_array, jnp.expand_dims(jnp.arange(time_steps), -1))

    def reconstruct(map_tree, net_state, model):
        x, y, time = map_tree
        cel, grads = loss_grad_fn(net_state, x, y, time, model)
        return cel, grads

    time_rec_map = jax.vmap(
        partial(reconstruct, net_state=net_state, model=model))
    mses, grads = time_rec_map(map_tree)
    time_update_map = jax.vmap(
        partial(opt.update, state=opt_state, params=net_state))
    updates, opt_states = time_update_map(grads)

    time_net_states_map = jax.vmap(
        partial(optax.apply_updates, params=net_state))
    net_states = time_net_states_map(updates=updates)

    net_state = jax.tree_map(partial(jnp.mean, axis=0), net_states)
    mean_opt_state = jax.tree_map(partial(jnp.mean, axis=0), opt_states)
    opt_state = jax.tree_map(lambda t, r: t.astype(r.dtype),
                             mean_opt_state,  opt_state)

    mse = jnp.mean(mses)
    return mse, net_state, opt_state


def test(net_state: FrozenDict, model: nn.Module, key: int,
         input_shape: List[int], time_steps: int):
    prng_key = jax.random.PRNGKey(key)
    process_array = jax.random.uniform(
        prng_key, [1] + input_shape,
        minval=-1, maxval=1)
    for time in range(time_steps):
        process_array = model.apply(net_state, 
            (process_array, jnp.expand_dims(jnp.array(time), -1)))[:, :, :, 0]
    return process_array


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    batch_size = args.batch_size
    gpus = args.gpus if args.gpus > 0 else jax.local_device_count() 

    print(f"Working with {gpus} gpus.")

    dataset_img, _ = get_mnist_train_data()
    print("Data loaded. Starting to train.")
    # train_data = np.stack([np.array(img) for img in dataset['train']['image']])
    print(f"Splitting {dataset_img.shape}, into {batch_size*gpus} parts. ")
    train_batches = np.array_split(
        dataset_img,
        len(dataset_img) // batch_size*gpus)

    input_shape = list(np.array(train_batches[0][0]).shape)
    stats = {"mean": jnp.array(np.mean(dataset_img)),
             "std": jnp.array(np.std(dataset_img))}

    model = UNet()
    opt = optax.adam(0.001)
    # create the model state
    net_state = model.init(key, 
            (jnp.ones([batch_size] + input_shape),
             jnp.array([1.])))
    opt_state = opt.init(net_state)
    iterations = 0

    # @partial(jax.jit, static_argnames= ['model', 'opt', 'time_steps'])
    def central_step(img: jnp.ndarray,
                     net_state: FrozenDict,
                     opt_state: FrozenDict,
                     model: nn.Module,
                     opt: optax.GradientTransformation,
                     time_steps: int):
        img = jnp.array(img)
        img_norm = (img - stats["mean"]) / stats["std"]
        print(f"Split {img.shape}, into {gpus}")
        if img.shape[0] % gpus != 0:
            img = img[:(img.shape[0]//gpus)*gpus]
            print(f"lost, images. New shape: {img.shape}")
        img_norm = jnp.stack(jnp.split(img, gpus))
        print(f"input shape: {img_norm.shape}")

        partial_train_step = partial(train_step,
        net_state=net_state, opt_state=opt_state,                          
        model=model, opt=opt, time_steps=time_steps)
        pmap_train_step = jax.pmap(
            partial_train_step, devices=jax.devices()[:gpus]
        )
        mses, net_states, opt_states = pmap_train_step(
            batch=img_norm,
            seed=jnp.expand_dims(jnp.array(args.seed)+jnp.arange(gpus), -1)
            )
        mean_loss = jnp.mean(mses)
        net_state = jax.tree_map(partial(jnp.mean, axis=0),
                                net_states)
        mean_opt_state = jax.tree_map(partial(jnp.mean, axis=0),
                                    opt_states)
        opt_state = jax.tree_map(lambda t, r: t.astype(r.dtype),
                                 mean_opt_state,  opt_states)
        return mean_loss, net_state, opt_state
            
    for e in range(args.epochs):
        for pos, img in enumerate(train_batches):
            mean_loss, net_state, opt_state = central_step(
                img, net_state, opt_state, model, opt,
                args.time_steps)
            if pos % 10 == 0:
                print(e, pos, mean_loss, len(train_batches))

            iterations += 1

    # testing...
    test_image = test(net_state, model, 5, input_shape, 10)
    plt.imshow(test_image[0])
    plt.savefig('test_img1.png')
    plt.show()
    test_image = test(net_state, model, 6, input_shape, 10)
    plt.imshow(test_image[0])
    plt.savefig('test_img2.png')
    plt.show()
    breakpoint()
