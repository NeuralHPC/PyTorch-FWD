from typing import List
import pytest
import jax
import jax.numpy as jnp

from src.Improved_UNet.UNet import Improv_UNet

input_shape = [
    [64, 64, 3]
]
@pytest.mark.parametrize("input_shape", input_shape)
def test_imporv_Unet(input_shape: List):
    """Test UNet from the Improved Denoising Diffusion Probabilistic Models (https://arxiv.org/abs/2102.09672)"""
    model = Improv_UNet(
        out_channels=input_shape[-1],
        model_channels=128,
        classes=1000,
        num_res_blocks=1,
        num_heads=2,
        num_heads_ups=2
    )
    key = jax.random.PRNGKey(42)
    batch_size = 10
    net_state = model.init(
        key,
        (jnp.ones([batch_size] + input_shape),
             jnp.ones([batch_size]),
             jnp.expand_dims(jnp.ones([batch_size]), -1))
    )
    dummy_input = (jnp.zeros([batch_size] + input_shape),
             jnp.ones([batch_size]),
             jnp.expand_dims(jnp.ones([batch_size]), -1))
    dummy_output = model.apply(net_state, dummy_input)
    assert dummy_output.shape == tuple([batch_size]+input_shape)
