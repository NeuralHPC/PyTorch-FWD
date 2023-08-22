"""Compute FID Score based on the sampled and original dataset."""

import argparse

import os
import jax
import jax.numpy as jnp
import flax.linen as nn
from src.fid import inception, fid
from functools import partial
import numpy as np
import glob
from tqdm import tqdm


def rescale(img):
    img = (img - jnp.min(img))/(jnp.max(img) - jnp.min(img))*255.0
    return img.astype(jnp.uint8)


parser = argparse.ArgumentParser(description='Compute FID score')
parser.add_argument(
    '--data-dir',
    required=True,
    help='Base dataset path'
)
parser.add_argument(
    '--sample-dir',
    required=True,
    help='Sampled images path'
)
parser.add_argument(
    '--input-size',
    type=int,
    required=True,
    help='Sampled input shape'
)
parser.add_argument(
    '--batch-size',
    type=int,
    default=25,
    help='Batch size for the InceptionV3 model'
)
if __name__ == '__main__':
    args = parser.parse_args()
    fname = f'data/CelebAHQ/stats_{args.input_size}x{args.input_size}.npz'
    sample_stats_fname = os.path.join(args.sample_dir, f"stats_{args.input_size}x{args.input_size}.npz")
    batch_size = args.batch_size

    model = inception.InceptionV3(pretrained=True)
    net_state = model.init(
        jax.random.PRNGKey(0),
        jnp.ones((1, 256, 256, 3))
    )
    apply_fn = jax.jit(partial(model.apply, train=False))

    mu_orig, mu_sampled = None, None
    sigma_orig, sigma_sampled = None, None

    # Compute mean and variance for original dataset if doesn't exist
    if not os.path.exists(fname):
        print("Mean and variance not found for original dataset, calculating...")
        mu_orig, sigma_orig = fid.compute_statistics(args.data_dir, net_state, apply_fn, batch_size, (256, 256))
        np.savez(fname, mu=mu_orig, sigma=sigma_orig)
        print('Compute and saved the raw data statistics')
    else:
        print("Found mean and variance for original data, loading...")
        with jnp.load(fname) as data:
            mu_orig = data['mu']
            sigma_orig = data['sigma']

    # Compute mean and variance for the sampled dataset if doesn't exist
    if not os.path.exists(sample_stats_fname):
        print("Mean and variance not found for sampled dataset, calculating...")
        sampled_imgs = []
        for npz_file in glob.glob(os.path.join(args.sample_dir, "sampled_imgs_*.npz")):
            with jnp.load(npz_file) as imgs:
                sampled_imgs.append(imgs['imgs'])
        sampled_imgs = jnp.concatenate(sampled_imgs, axis=0)
        sampled_imgs = jax.vmap(rescale)(sampled_imgs)
        
        batch_activations = []
        for idx in tqdm(range(len(sampled_imgs)//batch_size)):
            batch_imgs = sampled_imgs[idx*batch_size : (idx+1)*batch_size, :, :, :]
            acts = fid.compute_sampled_statistics(sampled_imgs, net_state, apply_fn)
            batch_activations.append(acts)
        batch_activations = jnp.concatenate(batch_activations, axis=0)
        mu_sampled = jnp.mean(batch_activations, axis=0)
        sigma_sampled = jnp.cov(batch_activations, rowvar=False)
        jnp.savez(sample_stats_fname, mu=mu_sampled, sigma=sigma_sampled)
    else:
        print("Found mean and variance for sampled data, loading...")
        with jnp.load(fname) as data:
            mu_sampled = data['mu']
            sigma_sampled = data['sigma']

    print("Computing the FID...")
    fid_score = fid.compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6)
    print(f'Fid Score: {fid_score}')
