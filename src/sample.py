from typing import List

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

from src.util import get_mnist_test_data

import matplotlib.pyplot as plt

def sample_noise(img: jnp.ndarray,
                 current_time_step: int,
                 key: jnp.ndarray,
                 max_steps: int):
    alpha = current_time_step/max_steps
    noise = jax.random.normal(
            key, shape=img.shape)
    x = (1-alpha)*img + noise * alpha
    return x, img - x


def sample_net_noise(net_state: FrozenDict, model: nn.Module, key: int,
         input_shape: List[int], max_steps: int):
    prng_key = jax.random.PRNGKey(key)
    process_array = jax.random.normal(
            prng_key, shape=[1] + input_shape)
    
    for time in reversed(range(max_steps)):
        # de_noise = model.apply(net_state,
        #     (process_array,
        #      jnp.expand_dims(jnp.array(time), -1),
        #      jnp.expand_dims(jnp.array([9]), 0)))[:, :, :, 0]
        de_noise = model.apply(net_state,
               (process_array,
                jnp.expand_dims(jnp.array(time), -1),
                jnp.expand_dims(jnp.array([9]), 0)))
        process_array += de_noise
        prng_key = jax.random.split(prng_key, 1)[0]
        process_array = sample_noise(process_array, time, prng_key, max_steps)[0]
    return process_array[0]

def sample_net_test(net_state: FrozenDict, model: nn.Module, key: int,
        test_time_step: int, max_steps: int):
    test_img, test_lbl = get_mnist_test_data()
    test_img, test_lbl = test_img[:5], test_lbl[:5]
    test_img = test_img/255.
    key = jax.random.PRNGKey(key)
    x, y = sample_noise(test_img, test_time_step, key, max_steps)
    yhat = model.apply(net_state, (
        x,
        jnp.expand_dims(jnp.array(test_time_step), -1),
        jnp.expand_dims(test_lbl, -1)))[..., 0]
    rec = x + yhat
    rec_mse = jnp.mean(rec**2)
    noise_mse = jnp.mean((y-yhat)**2)
    return rec, rec_mse, noise_mse