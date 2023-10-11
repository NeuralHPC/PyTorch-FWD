import sys

import jax
import jax.numpy as jnp
from scripts.train_diffuse_mnist import norm_and_split

from src.sample_util import sample_noise
from src.util import get_mnist_test_data

sys.path.append("../")


def test_denoise():
    data = get_mnist_test_data()[0]
    data = jnp.array(data[:5] / 255)
    key = jax.random.PRNGKey(1)
    max_steps = 10
    x, y = sample_noise(data, 8, key, max_steps)
    assert jnp.allclose((y + x), data)


def test_norm_and_split():
    # Test non-divisible case
    key = jax.random.PRNGKey(0)
    sample_imgs = jax.random.normal(key, (1622, 3, 3))
    sample_lbls = jax.random.normal(key, (1622, 1))
    gpus = 4
    split_imgs, split_lbls = norm_and_split(
        img=sample_imgs,
        lbl=sample_lbls,
        gpus=gpus
    )
    assert split_imgs.shape == (4, 405, 3, 3)
    assert split_lbls.shape == (4, 405, 1)
    # Test divisible case
    sample_imgs = jax.random.normal(key, (400, 3, 3))
    sample_lbls = jax.random.normal(key, (400, 1))
    gpus = 5
    split_imgs, split_lbls = norm_and_split(
        img=sample_imgs,
        lbl=sample_lbls,
        gpus=gpus
    )
    assert split_imgs.shape == (5, 80, 3, 3)
    assert split_lbls.shape == (5, 80, 1)
