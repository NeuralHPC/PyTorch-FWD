from typing import List, Tuple

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

from src.util import get_mnist_test_data, get_batched_celebA_paths, batch_loader

import matplotlib.pyplot as plt

def get_alpha(current_time_step, max_steps):
    betas = jnp.linspace(0.0001, 0.02, max_steps)
    alphas = 1-betas
    alpha_cumprod = jnp.cumprod(alphas)
    return alpha_cumprod[current_time_step]

def sample_noise(img: jnp.ndarray,
                 current_time_step: int,
                 key: jnp.ndarray,
                 max_steps: int):
    # alpha = current_time_step/max_steps
    alpha = get_alpha(current_time_step, max_steps)
    noise = jax.random.normal(
            key, shape=img.shape)
    # x = (1-alpha)*img + noise * alpha
    x = jnp.sqrt(alpha)*img + jnp.sqrt(1-alpha)*noise
    # return x, img - x
    return x, noise


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
                jnp.expand_dims(jnp.array([3338]), 0)))
        process_array -= de_noise
        prng_key = jax.random.split(prng_key, 1)[0]
        process_array = sample_noise(process_array, time, prng_key, max_steps)[0]
    return process_array[0]

def sample_net_test(net_state: FrozenDict, model: nn.Module, key: int,
        test_time_step: int, max_steps: int, data_dir: str):
    test_img, test_lbl = get_mnist_test_data(data_dir)
    test_img, test_lbl = test_img[:5], test_lbl[:5]
    test_img = jnp.expand_dims(test_img, -1)
    test_img = test_img/255.
    key = jax.random.PRNGKey(key)
    x, y = sample_noise(test_img, test_time_step, key, max_steps)
    yhat = model.apply(net_state, (
        x,
        jnp.expand_dims(jnp.array(test_time_step), -1),
        jnp.expand_dims(test_lbl, -1)))
    rec = x + yhat
    rec_mse = jnp.mean(rec**2)
    noise_mse = jnp.mean((y-yhat)**2)
    return rec, rec_mse, noise_mse


def sample_net_test_celebA(net_state: FrozenDict, model: nn.Module, key:int,
        test_time_step: int, max_steps: int, test_data: Tuple):
    key = jax.random.PRNGKey(key)
    test_img, test_lbl = test_data
    test_img = test_img/255.
    x, y = sample_noise(test_img, test_time_step, key, max_steps)
    y_hat = model.apply(net_state,(
        x,
        jnp.expand_dims(jnp.array(test_time_step), -1),
        jnp.expand_dims(test_lbl, -1)))
    rec = x - y_hat
    rec_mse = jnp.mean(rec**2)
    noise_mse = jnp.mean((y-y_hat)**2)
    return rec, rec_mse, noise_mse