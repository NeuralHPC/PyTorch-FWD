from typing import List
import pytest
import jax
import jax.numpy as jnp

from src.networks import UNet


input_shapes = [
    [128, 128, 3],
    [256, 256, 3]
]
@pytest.mark.parametrize("input_shape", input_shapes)
def test_UNet(input_shape: List):
    model = UNet(output_channels=input_shape[-1])
    key = jax.random.PRNGKey(42)
    batch_size = 1
    net_state = model.init(
        key,
        (jnp.ones([batch_size] + input_shape),
             jnp.expand_dims(jnp.ones([batch_size]), -1),
             jnp.expand_dims(jnp.ones([batch_size]), -1)))
    dummy_input = (jnp.zeros([batch_size] + input_shape),
             jnp.expand_dims(jnp.ones([batch_size]), -1),
             jnp.expand_dims(jnp.ones([batch_size]), -1))
    dummy_output = model.apply(net_state, dummy_input)
    assert dummy_output.shape == tuple([batch_size]+input_shape)
