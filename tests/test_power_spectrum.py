"""Tests for KLDivergence Power Spectrum."""

from copy import deepcopy
from itertools import pairwise
from typing import Tuple

import numpy as np
import pytest
import torch as th

from scripts.wpkl.wpkl import compute_packets, wavelet_power_divergence
from tests.test_wavelet_frechet_distance import (
    default_params,
    get_images,
    make_dataloader,
)

th.set_default_dtype(th.float64)
th.use_deterministic_algorithms(True)


def _calc_klwd(target_images: th.Tensor, output_images: th.Tensor) -> float:
    """Compute KLDivergence Power spectrum.

    Args:
        target_images (th.Tensor): Ground Truth images
        output_images (th.Tensor): Generated images

    Returns:
        float: WPKL value.
    """
    default_params["dataloader"] = make_dataloader(target_images)
    target_packets = compute_packets(**default_params)
    default_params["dataloader"] = make_dataloader(output_images)
    output_packets = compute_packets(**default_params)

    klwd = wavelet_power_divergence(target_packets, output_packets)
    return klwd


def test_same_input():
    """WPKL-test for same input."""
    target_images = get_images()
    output_images = deepcopy(target_images)
    klwd = _calc_klwd(target_images, output_images)
    assert np.allclose(klwd, 0.0)


def test_shuffle_input():
    """WPKL-test for shuffled input."""
    target_images = get_images()
    output_images = deepcopy(target_images)
    permutation = th.randperm((len(target_images)))
    shuffled_images = output_images[permutation, :, :, :]
    assert not th.allclose(shuffled_images, output_images)

    shuffled_klwd = _calc_klwd(target_images, shuffled_images)
    unshuffled_klwd = _calc_klwd(target_images, output_images)
    assert np.allclose(shuffled_klwd, unshuffled_klwd)
    assert np.allclose(shuffled_klwd, 0.0)
    assert np.allclose(unshuffled_klwd, 0.0)


@pytest.mark.slow
@pytest.mark.parametrize("wavelet", ["sym5", "db5", "Haar"])
@pytest.mark.parametrize("level", [2, 3])
def test_various_wavelets(wavelet: str, level: int):
    """WPKL-test for various wavelets and transformation levels.

    Args:
        wavelet (str): Type of wavelet. Accepted values [sym5, db5, Haar]
        level (int): Packet transformation level. Accepted values [2, 3].
    """
    target_images = get_images()
    output_images = deepcopy(target_images)

    default_params["wavelet"] = wavelet
    default_params["max_level"] = level

    fwd = _calc_klwd(target_images, output_images)
    assert np.allclose(fwd, 0.0, atol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("img_size_level", [(32, 1), (64, 2)])
def test_various_image_sizes(img_size_level: Tuple[int, int]):
    """Test various image sizes and corresponding transfromation levels.

    Args:
        img_size_level (Tuple[int, int]): Tuple containing img size and transfromation level.
    """
    size, level = img_size_level

    default_params["max_level"] = level

    target_images = get_images(size)
    output_images = deepcopy(target_images)
    assert output_images.shape == (8, 3, size, size)

    fwd = _calc_klwd(target_images, output_images)
    assert np.allclose(fwd, 0.0, atol=1e-3)


@pytest.mark.slow
def test_checkerboard_power_kldiv():
    """WPKL-Synthetic frequency test using various checkerboard sizes."""
    import scipy.ndimage

    range_max = 100
    # tile_size = 10 # determines tile size
    images = []

    for tile_size in [2, 5, 10]:
        # generate random grid

        grid = np.meshgrid(np.arange(0, range_max), np.arange(0, range_max))
        # grid = grid[0] + grid[1]

        tile_fun = lambda cord, ts: (cord % ts) < (ts // 2)  # type: ignore # noqa: E731

        x_tile = tile_fun(grid[0], tile_size)
        y_tile = tile_fun(grid[1], tile_size)
        grid = 1 - (x_tile + y_tile)

        grid = grid.astype(np.float32)

        # generate 10 images via a random fft-shift
        img_list = []
        for _ in range(1000):
            input_ = np.fft.fft2(grid)
            shift = np.random.uniform(-1, 1) * tile_size // 2
            image = scipy.ndimage.fourier_shift(input_, shift=shift)
            image = np.fft.ifft2(image)
            img_list.append(image.real)
        # B, H, W
        img_array = np.stack(img_list, axis=0)
        # B, C, H, W
        img_array = np.expand_dims(img_array, 1)
        # img_array = np.stack([img_array]*3, 1) # TODO!
        images.append(img_array)

    # a list for the wavelet frecet distances.
    wfd_list = []
    default_params["max_level"] = 3
    reference = images[0]
    for cimgs in images[1:]:
        kwd = _calc_klwd(
            target_images=th.from_numpy(reference), output_images=th.from_numpy(cimgs)
        )
        wfd_list.append(kwd)
    assert all(a < b for a, b in pairwise(wfd_list))
