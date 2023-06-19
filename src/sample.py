from typing import List

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

def sample_net(net_state: FrozenDict, model: nn.Module, key: int,
         input_shape: List[int], time_steps: int, bound: float=0.8):
    prng_key = jax.random.PRNGKey(key)
    noise_array = jax.random.uniform(
        prng_key, [time_steps] + input_shape,
        minval=-bound, maxval=bound)
    process_array = jnp.cumsum(noise_array, axis=0)

    for time in range(time_steps):
        process_array = jnp.clip(process_array, -1., 1.)
        process_array = model.apply(net_state, 
            (process_array, jnp.expand_dims(jnp.array(time), -1)))[:, :, :, 0]
    return process_array[0]

