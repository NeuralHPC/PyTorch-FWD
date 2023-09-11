"""Train diffusion model for CelebAHQ dataset."""

import datetime
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from clu import metric_writers, parameter_overview

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

import numpy as np

from src.util import _parse_args, get_batched_celebA_paths, batch_loader, _save_model, get_only_celebA_batches
from src.Improved_UNet.UNet import Improv_UNet
from src.sample import sample_noise, sample_net_noise, sample_net_test_celebA, sample_DDPM
from src.freq_math import forward_wavelet_packet_transform, inverse_wavelet_packet_transform

global_use_wavelet_cost = False
global_use_fourier_cost = False


@partial(jax.jit, static_argnames=['model'])
def diff_step(net_state, x, y, labels, time, model):
    denoise = model.apply(net_state, (x, time, labels))
    pixel_mse_cost = jnp.mean(0.5 * (y - denoise) ** 2)

    def packet_norm(packs):
        max_vals = jnp.max(abs(packs), axis=(0,1))
        packs /= max_vals
        return packs
    
    y_packets = packet_norm(forward_wavelet_packet_transform(y))
    net_packets = packet_norm(forward_wavelet_packet_transform(denoise))

    packet_mse_cost = jnp.mean(0.5 * (y_packets - net_packets) ** 2)

    if global_use_wavelet_cost:
        cost = 0.7 * pixel_mse_cost + 0.3 * packet_mse_cost
    else:
        cost = pixel_mse_cost

    return cost, (pixel_mse_cost, packet_mse_cost)


diff_step_grad = jax.value_and_grad(diff_step, argnums=0, has_aux=True)


def train_step(batch: jnp.ndarray,
               labels: jnp.ndarray,
               net_state: FrozenDict,
               seed: int, model: nn.Module,
               time_steps: int):
    key = jax.random.PRNGKey(seed[0])
    current_step_array = jax.random.randint(
        key, shape=[batch.shape[0]], minval=1, maxval=time_steps)
    seed_array = jax.random.split(key, batch.shape[0])
    batch_map = jax.vmap(partial(sample_noise, max_steps=time_steps))
    x, y = batch_map(batch, current_step_array, seed_array)

    cost_and_aux, grads = diff_step_grad(net_state, x, y, labels,
                                current_step_array, model)
    mse, freq_aux = cost_and_aux
    return mse, grads, freq_aux


def testing(e, net_state, model, input_shape, writer, time_steps, test_data):
    seed = 5
    time_steps = 100
    test_image = sample_net_noise(net_state, model, seed, input_shape, time_steps)
    time_steps_list = [1, 10, 25, 50]
    writer.write_images(e, {
        f'fullnoise_{time_steps}_{seed}': test_image})

    for test_time in time_steps_list:
        test_image, rec_mse, _ = sample_net_test_celebA(net_state, model, seed, test_time, time_steps, test_data)
        writer.write_images(e, {f'test_{test_time}_{seed}': test_image})
        writer.write_scalars(e, {f'test_rec_mse_{test_time}_{seed}': rec_mse})
        
    seed = 6
    test_image, _ = sample_DDPM(net_state, model, seed, input_shape, time_steps)
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
    print("Running diffusion training on celebA.") 
    args = _parse_args()
    print(args)
    
    global global_use_wavelet_cost
    global_use_wavelet_cost = args.wavelet_loss
    print(f"Using wavelet loss: {global_use_wavelet_cost}")

    now = datetime.datetime.now()
    
    # create the logdir if it does not exist already.
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    writer = metric_writers.create_default_writer(args.logdir + f"/weighted_pixel_packetnorm_{now}")
    checkpoint_dir = os.path.join(args.logdir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    
    batch_size = args.batch_size
    gpus = args.gpus if args.gpus > 0 else jax.local_device_count() 

    print(f"Working with {gpus} gpus.")

    resize = None
    if args.resize:
        resize = (args.resize, args.resize)

    train_batched_fn = None
    test_batched_fn = None
    if args.dataset.lower() == "celebahq":
        print("Loading CelebAHQ dataset")
        train_batched_fn = test_batched_fn = get_batched_celebA_paths
    elif args.dataset.lower() == "celeba":
        print("Loading CelebA dataset")
        train_batched_fn = get_only_celebA_batches
        test_batched_fn = partial(get_only_celebA_batches, split='validation')
    else:
        raise NotImplementedError("Only supported datasets are CelebAHQ and CelebA.")

    path_batches, labels_dict = train_batched_fn(args.data_dir, batch_size)
    print("Data loaded. Starting to train")
    dummy_img_batch, _ = batch_loader(path_batches[0], labels_dict, resize)
    input_shape = list(dummy_img_batch[0].shape)
    print(f"Total {len(path_batches)} number of batches")
    batch_loader_w_dict = partial(batch_loader, labels_dict=labels_dict, resize=resize)
    print(f"Input shape: {input_shape}")

    test_patches, labels_dict = test_batched_fn(args.data_dir, batch_size)
    imgs, labels = batch_loader(test_patches[0], labels_dict, resize)
    test_data = (imgs[:5], labels[:5])

    # Process model related args
    if args.conditional:
        print("Using class conditional")
    
    out_channels = input_shape[-1]

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
        out_channels=out_channels,
        base_channels=args.base_channels,
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
                net_state=net_state, model=model,
                time_steps=time_steps)
        pmap_train_step = jax.pmap(
             partial_train_step, devices=jax.devices()[:gpus]
        )
        mses, multi_grads, freq_aux = pmap_train_step(
            batch=img_norm, labels=lbl,
            seed=jnp.expand_dims(jnp.array(seed)+jnp.arange(gpus), -1)
            )
        mean_loss = jnp.mean(mses)
        freq_loss_mean = jnp.mean(jnp.array(freq_aux), axis=1)
        grads = jax.tree_map(partial(jnp.mean, axis=0), multi_grads) # Average the gradients
        updates, opt_state = opt.update(grads, opt_state, net_state)
        net_state = optax.apply_updates(net_state, updates)
        return mean_loss, net_state, opt_state, freq_loss_mean

    for e in range(args.epochs):
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            load_asinc_dict = {executor.submit(batch_loader_w_dict, path_batch): path_batch
                            for path_batch in path_batches}
            for pos, future_train_batches in enumerate(as_completed(load_asinc_dict)):
                img, lbl = future_train_batches.result()
                lbl = jnp.expand_dims(lbl, -1)
                mean_loss, net_state, opt_state, freq_aux = central_step(
                    img, lbl, net_state, opt_state, iterations*gpus,
                    model, opt, args.time_steps)
                if pos % 50 == 0:
                    print(e, pos, mean_loss, len(load_asinc_dict))

                iterations += 1
                writer.write_scalars(iterations, {"loss": mean_loss})
                pixel_mse_cost, packet_mse_cost = freq_aux
                writer.write_scalars(iterations, {"pixel_mse_cost": jnp.mean(pixel_mse_cost)})
                writer.write_scalars(iterations, {"packet_mse_cost": jnp.mean(packet_mse_cost)})

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
