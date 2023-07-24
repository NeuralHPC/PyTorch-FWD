from typing import List

import pickle
import jax
import jax.numpy as jnp

from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
from src.sample import sample_net_test, sample_noise
from src.util import write_movie

import matplotlib.pyplot as plt


def sample_net_noise(net_state: FrozenDict, model: nn.Module, key: int,
         input_shape: List[int], max_steps: int):
    prng_key = jax.random.PRNGKey(key)
    process_array = jax.random.normal(
            prng_key, shape=[1] + input_shape)
    steps = [process_array]
    
    for time in reversed(range(max_steps)):
        de_noise = model.apply(net_state, 
            (process_array,
             jnp.expand_dims(jnp.array(time), -1),
             jnp.expand_dims(jnp.array([9]), 0)))[:, :, :, 0]
        process_array += de_noise
        prng_key = jax.random.split(prng_key, 1)[0]
        steps.append(process_array)
        process_array = sample_noise(process_array, time, prng_key, max_steps)[0]
    return process_array[0], steps




if __name__ == '__main__':
    # checkpoint_path = "/home/wolter/uni/diffusion/log/checkpoints/e_5_time_2023-06-22 12:35:07.968094.pkl"
    checkpoint_path = '/home/wolter/tunnel/infgpu/uni/diffusion/log/checkpoints/e_390_time_2023-06-22 16:27:04.801459.pkl'

    with open(checkpoint_path, 'rb') as f:
        loaded = pickle.load(f)

    (net_state, opt_state, model) = loaded
    test_img, steps = sample_net_noise(net_state, model, 4, [28, 28], 40)
    write_movie([s[0] for s in steps], xlim=1, ylim=1)
    plt.imshow((test_img + 1)/2.)
    plt.show()


    rec1 = sample_net_test(net_state, model, 2, 1)
    rec5 = sample_net_test(net_state, model, 2, 5)
    rec10 = sample_net_test(net_state, model, 2, 10)
    rec20 = sample_net_test(net_state, model, 2, 20)
    rec30 = sample_net_test(net_state, model, 2, 30)
    rec40 = sample_net_test(net_state, model, 2, 40)
    pass