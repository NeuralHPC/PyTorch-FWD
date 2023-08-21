from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import random

import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

from src.util import get_mnist_test_data


def linear_noise_scheduler(current_time_step: int, max_steps: int) -> Tuple[jnp.ndarray]:
    """Sample linear noise scheduler.

    Args:
        current_time_step (int): Current time
        max_steps (int): Maximum number of steps

    Returns:
        Tuple[jnp.ndarray]: Tuple containing alpha_bar, alpha and betas at current time
    """
    betas = jnp.linspace(0.0001, 0.02, max_steps)
    alphas = 1-betas
    alpha_ts = jnp.cumprod(alphas)
    return alpha_ts[current_time_step], alphas[current_time_step], betas[current_time_step]


def sample_noise(img: jnp.ndarray,
                 current_time_step: int,
                 key: jnp.ndarray,
                 max_steps: int) -> Tuple[jnp.ndarray]:
    """Diffusion forward step.

    Args:
        img (jnp.ndarray): Input image
        current_time_step (int): Current time
        key (jnp.ndarray): PRNGKey
        max_steps (int): Total time

    Returns:
        Tuple[jnp.ndarray]: Tuple containing the noised image and noise
    """
    alpha_t, _, _ = linear_noise_scheduler(current_time_step, max_steps)
    noise = jax.random.normal(
            key, shape=img.shape)
    x = jnp.sqrt(alpha_t)*img + jnp.sqrt(1-alpha_t )*noise
    return x, noise


def sample_net_noise(net_state: FrozenDict, model: nn.Module, key: int,
         input_shape: List[int], max_steps: int):
    prng_key = jax.random.PRNGKey(key)
    process_array = jax.random.normal(
            prng_key, shape=[1] + input_shape)
    
    for time in reversed(range(max_steps)):
        de_noise = model.apply(net_state,
               (process_array,
                jnp.expand_dims(jnp.array(time), -1),
                jnp.expand_dims(jnp.array([3338]), 0)))
        process_array -= de_noise
        prng_key = jax.random.split(prng_key, 1)[0]
        process_array = sample_noise(process_array, time, prng_key, max_steps)[0]
    return process_array[0]


def sample_DDPM(net_state: FrozenDict, model: nn.Module, key: int,
                    input_shape: List[int], max_steps: int, test_label: int = 3338) -> Union[np.ndarray, List[np.ndarray]]:
    """DDPM Sampling from https://arxiv.org/pdf/2006.11239.pdf.

    Args:
        net_state (FrozenDict): Model parameters.
        model (nn.Module): Model instance.
        key (int): PRNGKey.
        input_shape (List[int]): input_shape.
        max_steps (int): Maximum steps.
        test_label (int, optional): Test label to sample for class conditioning. Defaults to 3338.

    Returns:
        Union[np.ndarray, List[np.ndarray]]: Return sampled image and all the steps.
    """
    if key == -1:
        key = random.randint(0, 50000)
    prng_key = jax.random.PRNGKey(key)
    x_t = jax.random.normal(
        prng_key, shape=[1]+input_shape
    )
    x_t_1 = x_t
    steps = [x_t_1]
    for time in reversed(range(max_steps)):
        alpha_t, alpha, _ = linear_noise_scheduler(time, max_steps)
        prng_key = jax.random.PRNGKey(random.randint(0, 50000))
        z = jax.random.normal(
            prng_key, shape=[1]+input_shape
        ) 
        denoise = model.apply(net_state,
                              (x_t_1,
                               jnp.expand_dims(jnp.array(time), -1),
                               jnp.expand_dims(jnp.array(test_label), 0)))
        x_mean = (x_t_1  - (denoise *((1-alpha)/(jnp.sqrt(1-alpha_t)))))/(jnp.sqrt(alpha))
        x_t_1 = x_mean + jnp.sqrt(1-alpha) * z
        steps.append(x_t_1)
    x_0 = x_t_1 - jnp.sqrt(1-alpha) * z
    return x_0[0], steps


def batch_DDPM(net_state: FrozenDict, model: nn.Module, key: int,
                    input_shape: List[int], max_steps: int, batch_size: int, test_label: List[int]) -> Union[np.ndarray, List[np.ndarray]]:
    """Batch DDPM Sampling from https://arxiv.org/pdf/2006.11239.pdf.

    Args:
        net_state (FrozenDict): Model parameters.
        model (nn.Module): Model instance.
        key (int): PRNGKey.
        input_shape (List[int]): input_shape.
        max_steps (int): Maximum steps.
        batch_size (int): Number of images to be sampled
        test_label (List[int]): Test labels to sample for class conditioning.

    Returns:
        Union[np.ndarray, List[np.ndarray]]: Return sampled image and all the steps.
    """
    if key == -1:
        key = random.randint(0, 50000)
    prng_key = jax.random.PRNGKey(key)
    x_t = jax.random.normal(
        prng_key, shape=[batch_size]+input_shape
    )
    x_t_1 = x_t
    # Lazy import
    from tqdm.auto import tqdm
    time_indices = tqdm(reversed(range(max_steps)), total=max_steps)
    for time in time_indices:
        alpha_t, alpha, _ = linear_noise_scheduler(time, max_steps)
        prng_key = jax.random.PRNGKey(random.randint(0, 50000))
        z = jax.random.normal(
            prng_key, shape=[batch_size]+input_shape
        )
        denoise = model.apply(net_state,
                              (x_t_1,
                               jnp.array([time]*batch_size),
                               test_label))
        x_mean = (x_t_1  - (denoise *((1-alpha)/(jnp.sqrt(1-alpha_t)))))/(jnp.sqrt(alpha))
        x_t_1 = x_mean + jnp.sqrt(1-alpha) * z
    return x_t_1



def sample_DDIM(net_state: FrozenDict, model: nn.Module, key: int,
                input_shape: List[int], max_steps: int, test_label: int = 3338) -> jnp.ndarray:
    """DDIM Sampling from https://arxiv.org/pdf/2010.02502.pdf.

    Args:
        net_state (FrozenDict): Model parameters.
        model (nn.Module): Model instance.
        key (int): PRNGKey.
        input_shape (List[int]): input_shape.
        max_steps (int): Maximum steps.
        test_label (int, optional): Test labels to sample for class conditioning.

    Returns:
        np.ndarray: Return the sampled image.
    """
    raise NotImplementedError


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