import datetime
from typing import List, Tuple
from functools import partial

from clu import metric_writers

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
    base_feat_no = 32 # 128

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
            y = y + time_emb
            y = nn.relu(nn.Conv(
                features=feats, kernel_size=(3, 3), padding="SAME")(y))
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


@partial(jax.jit, static_argnames=['model'])
def diff_step(net_state, x, y, time, model):
    denoise = model.apply(net_state, (x, time))
    cost = jnp.mean(0.5 * (jnp.expand_dims(y, -1) - denoise) ** 2)
    return cost

diff_step_grad = jax.value_and_grad(diff_step, argnums=0)


# @partial(jax.jit, static_argnames=['model', 'opt', 'time_steps'])
def train_step(batch: jnp.ndarray,
               net_state: FrozenDict, opt_state: FrozenDict,
               seed: int, model: nn.Module,
               opt: optax.GradientTransformation,
               time_steps: int):
    batch = batch
    key = jax.random.PRNGKey(seed[0])
    noise_array = jax.random.uniform(
        key, [time_steps] + list(batch.shape),
        minval=-.8, maxval=.8)
    cum_noise_array = jnp.cumsum(noise_array, axis=0)

    x_array = jnp.expand_dims(batch, 0) + cum_noise_array
    y_array = jnp.expand_dims(batch, 0) + jnp.concatenate(
        [jnp.zeros([1] + list(batch.shape)), cum_noise_array[:-1]])
    
    x_array = jnp.clip(x_array, -1., 1.)
    y_array = jnp.clip(y_array, -1., 1.)
    time = jnp.expand_dims(jnp.arange(time_steps), -1)
    map_tree = (x_array, y_array, time)

    def reconstruct(map_tree, net_state, model):
        x, y, time = map_tree
        cel, grads = diff_step_grad(net_state, x, y, time, model)
        return cel, grads

    time_rec_map = jax.vmap(
        partial(reconstruct, net_state=net_state, model=model))
    mses, grads = time_rec_map(map_tree)

    def update(net_state, opt_state, grads):
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
        return net_state, opt_state
    
    net_state, opt_state = update(net_state, opt_state, grads)

    # recurrent diffusion update
    # time_apply = jax.vmap(partial(model.apply, variables=net_state))
    # rec_x = time_apply(x_in=(x_array, time))
    # map_tree = (rec_x[:], y_array, time)
    # TODO: Finish

    mse = jnp.mean(mses)
    return mse, net_state, opt_state


def test(net_state: FrozenDict, model: nn.Module, key: int,
         input_shape: List[int], time_steps: int):
    prng_key = jax.random.PRNGKey(key)
    process_array = jax.random.uniform(
        prng_key, [1] + input_shape,
        minval=-.8, maxval=.8)
    for time in range(time_steps):
        process_array = jnp.clip(process_array, -1., 1.)
        process_array = model.apply(net_state, 
            (process_array, jnp.expand_dims(jnp.array(time), -1)))[:, :, :, 0]
        
    return process_array[0]


def testing(e, net_state, model, input_shape):
    seed = 5
    test_image = test(net_state, model, seed, input_shape, 30)
    plt.imshow(test_image)
    now = datetime.datetime.now()
    plt.savefig(f'out_img/{e}_{now}_test_{seed}.png')
    plt.clf()
    # writer.write_images(e, {
    #     'seed': jnp.expand_dims(jnp.expand_dims(test_image,0), -1)})
    seed = 6
    test_image = test(net_state, model, seed, input_shape, 30)
    plt.imshow(test_image)
    plt.savefig(f'out_img/{e}_{now}_test_{seed}.png')
    plt.clf()
    # writer.write_images(e, {
    #     'seed': jnp.expand_dims(jnp.expand_dims(test_image,0), -1)})


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    writer = metric_writers.create_default_writer(args.logdir)

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
        len(dataset_img) // (batch_size*gpus))

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

    @partial(jax.jit, static_argnames= ['model', 'opt', 'time_steps'])
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
        img_norm = jnp.stack(jnp.split(img_norm, gpus))
        print(f"input shape: {img_norm.shape}")

        partial_train_step = partial(train_step,
        net_state=net_state, opt_state=opt_state,                          
        model=model, opt=opt, time_steps=time_steps)
        pmap_train_step = jax.pmap(
             partial_train_step, devices=jax.devices()[:gpus]
        )
        # debug without jit
        # res = list(map(partial(
        #     partial_train_step,
        #     seed=jnp.array(args.seed)+jnp.arange(gpus)),
        #     img_norm))
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
            if pos % 50 == 0:
                print(e, pos, mean_loss, len(train_batches))

            iterations += 1
            writer.write_scalars(iterations, {"train_loss": mean_loss})

        if e % 5 == 0:
            print('testing...')
            testing(e, net_state, model, input_shape, writer)

    testing(e, net_state, model, input_shape, writer)
