import datetime
import pickle
from typing import List, Dict
from functools import partial
from multiprocess import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from clu import metric_writers, parameter_overview

import jax
import jax.numpy as jnp
# jax.config.update('jax_threefry_partitionable', True)

import optax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

import numpy as np
import matplotlib.pyplot as plt


from src.util import _parse_args, get_batched_celebA_paths, batch_loader, _save_model
# from src.networks import UNet
from src.Improved_UNet.UNet import Improv_UNet
from src.sample import sample_noise, sample_net_noise, sample_net_test_celebA, sample_DDPM



@partial(jax.jit, static_argnames=['model'])
def diff_step(net_state, x, y, labels, time, model):
    denoise = model.apply(net_state, (x, time, labels))
    cost = jnp.mean(0.5 * (y - denoise) ** 2)
    return cost

diff_step_grad = jax.value_and_grad(diff_step, argnums=0)


def train_step(batch: jnp.ndarray,
               labels: jnp.ndarray,
               net_state: FrozenDict,
               seed: int, model: nn.Module,
               time_steps: int):
    key = jax.random.PRNGKey(seed[0])
    
    current_step_array = jax.random.randint(
        key, shape=[batch.shape[0]], minval=1, maxval=time_steps)
    # current_step_array = jnp.expand_dims(current_step_array, -1)
    seed_array = jax.random.split(key, batch.shape[0])
    batch_map = jax.vmap(partial(sample_noise, max_steps=time_steps))
    x, y = batch_map(batch, current_step_array, seed_array)

    mse, grads = diff_step_grad(net_state, x, y, labels,
                                current_step_array, model)
    return mse, grads


def testing(e, net_state, model, input_shape, writer, time_steps, test_data):
    seed = 5
    time_steps = 100
    test_image = sample_net_noise(net_state, model, seed, input_shape, time_steps)
    # time_steps_list = [1, time_steps//4, time_steps//2]
    time_steps_list = [1, 10, 25, 50]
    writer.write_images(e, {
        f'fullnoise_{time_steps}_{seed}': test_image})
    # step tests
    for test_time in time_steps_list:
        test_image, rec_mse, _ = sample_net_test_celebA(net_state, model, seed, test_time, time_steps, test_data)
        writer.write_images(e, {f'test_{test_time}_{seed}': test_image})
        writer.write_scalars(e, {f'test_rec_mse_{test_time}_{seed}': rec_mse})
        
    seed = 6
    test_image = sample_DDPM(net_state, model, seed, input_shape, time_steps)
    writer.write_images(e, {
        f'fullnoise_{time_steps}_{seed}': test_image})
    for test_time in time_steps_list:
        test_image, rec_mse, _ = sample_net_test_celebA(net_state, model, seed, test_time, time_steps, test_data)
        writer.write_images(e, {
            f'test_{test_time}_{seed}': test_image})
        writer.write_scalars(e, {f'test_rec_mse_{test_time}_{seed}': rec_mse})


@partial(jax.jit, static_argnames='gpus')
def norm_and_split(img: jnp.ndarray,
              lbl: jnp.ndarray,
              gpus: int):
    print(f"Split {img.shape}, into {gpus}")
    if img.shape[0] % gpus != 0:
        img = img[:(img.shape[0]//gpus)*gpus]
        lbl = lbl[:(img.shape[0]//gpus)*gpus]
        print(f"lost, images. New shape: {img.shape}")
    img_norm = img / 255.
    img_norm = jnp.stack(jnp.split(img_norm, gpus))
    lbls = jnp.stack(jnp.split(lbl, gpus))
    print(f"input shape: {img_norm.shape}")
    return img_norm, lbls



def main():
    args = _parse_args()
    print(args)
    
    now = datetime.datetime.now()
    
    writer = metric_writers.create_default_writer(args.logdir + f"/{now}")
    checkpoint_dir = os.path.join(args.logdir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    
    batch_size = args.batch_size
    gpus = args.gpus if args.gpus > 0 else jax.local_device_count() 

    print(f"Working with {gpus} gpus.")

    path_batches, labels_dict = get_batched_celebA_paths(args.data_dir, batch_size)

    resize=None
    if args.resize:
        resize = (args.resize, args.resize)

    print("Data loaded. Starting to train.")
    dummy_img_batch, _ = batch_loader(path_batches[0], labels_dict, resize)
    input_shape = list(dummy_img_batch[0].shape)
    print(f"Total {len(path_batches)} number of batches")
    batch_loader_w_dict = partial(batch_loader, labels_dict=labels_dict, resize=resize)
    print(f"Input shape: {input_shape}")

    # Load test data images
    test_patches, labels_dict = get_batched_celebA_paths(args.data_dir)
    imgs, labels = batch_loader(test_patches[0], labels_dict, resize)
    test_data = (imgs[:5], labels[:5])

    # Process model related args
    if args.conditional:
        print("Using class conditional")
    
    if args.attn_heads_upsample == -1:
        args.attn_heads_upsample = args.attn_heads
    
    channel_mult = []
    for value in args.channel_mult.split(","):
        channel_mult.append(int(value))

    attn_res = []
    for value in args.attn_resolution.split(","):
        attn_res.append(input_shape[0]//int(value)) 

    # model = UNet(output_channels=input_shape[-1])
    model = Improv_UNet(
        out_channels=input_shape[-1],
        model_channels=args.base_channels,
        classes=args.conditional,
        channel_mult=tuple(channel_mult),
        num_res_blocks=args.num_res_blocks,
        num_heads=args.attn_heads,
        num_heads_ups=args.attn_heads_upsample,
        attention_res=tuple(attn_res)
    )
    opt = optax.adam(args.learning_rate)

    # create the model state
    net_state = model.init(key, 
            (jnp.ones([batch_size] + input_shape),
             jnp.ones([batch_size]),
             jnp.expand_dims(jnp.ones([batch_size]), -1)))
    print(parameter_overview.get_parameter_overview(net_state))
    opt_state = opt.init(net_state)
    iterations = 0

    @partial(jax.jit, static_argnames= ['model', 'opt', 'time_steps'])
    def central_step(img: jnp.ndarray,
                     lbl: jnp.ndarray,
                     net_state: FrozenDict,
                     opt_state: FrozenDict,
                     seed_offset: int,
                     model: nn.Module,
                     opt: optax.GradientTransformation,
                     time_steps: int):
        seed = args.seed + seed_offset
        img = jnp.array(img)
        img_norm, lbl = norm_and_split(img, lbl, gpus)
        partial_train_step = partial(train_step,
        net_state=net_state, model=model, time_steps=time_steps)
        pmap_train_step = jax.pmap(
             partial_train_step, devices=jax.devices()[:gpus]
        )
        mses, multi_grads = pmap_train_step(
            batch=img_norm, labels=lbl,
            seed=jnp.expand_dims(jnp.array(seed)+jnp.arange(gpus), -1)
            )
        mean_loss = jnp.mean(mses)
        grads = jax.tree_map(partial(jnp.mean, axis=0), multi_grads) # Average the gradients
        updates, opt_state = opt.update(grads, opt_state, net_state)
        net_state = optax.apply_updates(net_state, updates)
        return mean_loss, net_state, opt_state

    for e in range(args.epochs):
        with ThreadPoolExecutor() as executor:
            load_asinc_dict = {executor.submit(batch_loader_w_dict, path_batch): path_batch
                            for path_batch in path_batches}
            for pos, future_train_batches in enumerate(as_completed(load_asinc_dict)):
                img, lbl = future_train_batches.result()
                lbl = jnp.expand_dims(lbl, -1)
                mean_loss, net_state, opt_state = central_step(
                    img, lbl, net_state, opt_state, iterations*gpus,
                    model, opt, args.time_steps)
                if pos % 50 == 0:
                    print(e, pos, mean_loss, len(load_asinc_dict))

                iterations += 1
                writer.write_scalars(iterations, {"loss": mean_loss})
            print(' ', flush=True)
            if e % 5 == 0:
                print('testing...')
                testing(e, net_state, model, input_shape, writer,
                        time_steps=args.time_steps, test_data=test_data)
                to_storage = (net_state, opt_state, model)
                _save_model(checkpoint_dir, now, e, to_storage)

    print('testing...')
    testing(e, net_state, model, input_shape, writer, time_steps=args.time_steps, test_data=test_data)
    _save_model(checkpoint_dir, now, args.epochs, (net_state, opt_state, model))

if __name__ == '__main__':
    main()