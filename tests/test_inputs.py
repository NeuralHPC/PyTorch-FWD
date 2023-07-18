import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.sample import sample_noise
from src.util import get_mnist_test_data

sys.path.append("../")
import matplotlib.pyplot as plt

def test_denoise():
    data = get_mnist_test_data()[0]
    data = jnp.array(data[:5]/255)
    key = jax.random.PRNGKey(1)
    max_steps = 10
    x, y = sample_noise(data, 8, key, max_steps)
    assert jnp.allclose((y+x) , data)
