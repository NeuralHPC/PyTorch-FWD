import argparse

import os
import jax
import jax.numpy as jnp
import flax.linen as nn
from src.fid import inception, fid
from functools import partial
import numpy as np


parser = argparse.ArgumentParser(description='Compute FID score')
parser.add_argument(
    '--data-dir',
    required=True,
    help='Base dataset path'
)
parser.add_argument(
    '--sample-dir',
    required=True,
    help='Sample dataset path'
)
parser.add_argument(
    '--input-size',
    type=int,
    required=True,
    help='Sampled input shape'
)
if __name__ == '__main__':
    args = parser.parse_args()
    fname = f'data/CelebAHQ/sample_{args.input_size}x{args.input_size}.npz'
    orig_sample_exists = os.path.exists(fname)

    if not orig_sample_exists:
        model = inception.InceptionV3(pretrained=True)
        net_state = model.init(
            jax.random.PRNGKey(0),
            jnp.ones((1, 256, 256, 3))
        )
        batch_size = 50
        apply_fn = jax.jit(partial(model.apply, train=False))
        mu1, sigma1 = fid.compute_statistics(args.data_dir, net_state, apply_fn, batch_size, (256, 256))
        np.savez(fname, mu=mu1, sigma=sigma1)
        print('Compute and saved the raw data statistics')

    print('Loading statistics for CelebAHQ dataset')
    with jnp.load(fname) as data:
        mu1 = data['mu']
        sigma1 = data['sigma']
    
    print('Loading statistics for sampled dataset')
    with jnp.load(args.sample_dir) as data:
        mu2 = data['mu']
        sigma2 = data['sigma']

    fid_score = fid.compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6)
    print(f'Fid Score: {fid_score}')
