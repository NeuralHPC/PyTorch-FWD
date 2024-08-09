"""Test Frechet Inception distance."""

from copy import deepcopy
from typing import Tuple

import numpy as np
import pytest
import torch as th
from torch.nn.functional import adaptive_avg_pool2d

from pytorchfwd.freq_math import calculate_frechet_distance
from scripts.fid.inception import InceptionV3

from .test_wavelet_frechet_distance import get_images


@pytest.mark.slow
def forward_pass(images: th.Tensor) -> Tuple[np.ndarray, ...]:
    """Compute InceptionV3 forward pass.

    Args:
        images (th.Tensor): Image tensor.

    Returns:
        Tuple: Tuple containing mu and sigma of predictions.
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])
    with th.no_grad():
        pred = model(images)[0]
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    pred = pred.squeeze(3).squeeze(2).cpu().numpy()
    assert pred.shape == (images.shape[0], 2048)
    mu = np.mean(pred, axis=0)
    sigma = np.cov(pred, rowvar=False)
    return mu, sigma


@pytest.mark.slow
def test_same_input():
    """FID-test same input."""
    tensor_images = get_images(img_size=256)
    output_images = deepcopy(tensor_images)
    mu1, sigma1 = forward_pass(tensor_images)
    mu2, sigma2 = forward_pass(output_images)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    assert np.allclose(fid, 0.0, atol=1e-3)


@pytest.mark.slow
def test_shuffle_input():
    """FID-test shuffled input."""
    tensor_images = get_images(img_size=256)
    output_images = deepcopy(tensor_images)
    permutation = th.randperm(len(output_images))
    shuffled_images = output_images[permutation, :, :, :]
    assert not th.allclose(shuffled_images, output_images)
    mu_orig, sigma_orig = forward_pass(tensor_images)
    mu_shuff, sigma_shuff = forward_pass(shuffled_images)
    mu_op, sigma_op = forward_pass(output_images)
    original_fid = calculate_frechet_distance(mu_orig, sigma_orig, mu_op, sigma_op)
    shuffled_fid = calculate_frechet_distance(
        mu_orig, sigma_orig, mu_shuff, sigma_shuff
    )
    assert np.allclose(shuffled_fid, original_fid, atol=1e-4)
