from itertools import pairwise

import pytest
import scipy.datasets as datasets
import torch as th
import numpy as np
from copy import deepcopy
from torchvision.transforms import functional as tvf

from src.klwd import compute_packets, wavelet_power_divergence
from tests.test_wavelet_frechet_distance import get_images, make_dataloader, default_params

th.set_default_dtype(th.float64)
th.use_deterministic_algorithms(True)


def compute_klwd(target_images, output_images):
    default_params['dataloader'] = make_dataloader(target_images)
    target_packets = compute_packets(**default_params)
    default_params['dataloader'] = make_dataloader(output_images)
    output_packets = compute_packets(**default_params)

    klwd = wavelet_power_divergence(target_packets, output_packets)
    return klwd


@pytest.mark.slow
def test_same_input():
    target_images = get_images()
    output_images = deepcopy(target_images)
    klwd = compute_klwd(target_images, output_images)
    assert np.allclose(klwd, 0.)


@pytest.mark.slow
def test_shuffle_input():
    target_images = get_images(128)
    output_images = deepcopy(target_images)
    permutation = th.randperm((len(target_images)))
    shuffled_images = output_images[permutation, :, :, :]
    assert not th.allclose(shuffled_images, output_images)

    shuffled_klwd = compute_klwd(target_images, shuffled_images)
    unshuffled_klwd = compute_klwd(target_images, output_images)
    assert np.allclose(shuffled_klwd, unshuffled_klwd)
    assert np.allclose(shuffled_klwd, 0.)
    assert np.allclose(unshuffled_klwd, 0.)


@pytest.mark.slow
@pytest.mark.parametrize("wavelet", ["sym5", "db5", "Haar"])
@pytest.mark.parametrize("level", [2, 3, 4])
def test_various_wavelets(wavelet, level):
    target_images = get_images()
    output_images = deepcopy(target_images)

    default_params['wavelet'] = wavelet
    default_params['max_level'] = level

    fwd = compute_klwd(target_images, output_images)
    assert np.allclose(fwd, 0.0, atol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("img_size_level", [(32, 1), (64, 2), (128, 3)]) #  (256, 4)
def test_various_image_sizes(img_size_level):
    size, level = img_size_level

    default_params['max_level'] = level

    target_images = get_images(size)
    output_images = deepcopy(target_images)
    assert output_images.shape == (8, 3, size, size)
    
    fwd = compute_klwd(target_images, output_images)
    assert np.allclose(fwd, 0.0, atol=1e-3)


def test_checkerboard_power_KL():
    import scipy.ndimage

    range_max = 100 
    # tile_size = 10 # determines tile size
    images = []

    for tile_size in [2, 5, 10]:
    # generate random grid
    
        grid = np.meshgrid(np.arange(0,range_max), np.arange(0,range_max))
        # grid = grid[0] + grid[1]

        tile_fun = lambda cord: (cord % tile_size) < (tile_size // 2)

        x_tile = tile_fun(grid[0])
        y_tile = tile_fun(grid[1])
        grid = 1 - (x_tile + y_tile)

        grid = grid.astype(np.float32)

        # generate 10 images via a random fft-shift
        img_list = []
        for _ in range(1000):
            input_ = np.fft.fft2(grid)
            shift = np.random.uniform(-1, 1)*tile_size//2
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
    reference = images[0]
    for cimgs in images[1:]:
        kwd = compute_klwd(target_images=th.from_numpy(reference),
                           output_images=th.from_numpy(cimgs))
        wfd_list.append(kwd)
    assert all(a < b for a, b in pairwise(wfd_list))


@pytest.mark.slow
def test_gaussian_blur():
    target_images = get_images(128)
    blurred_images = []
    for kernel in (3, 5, 7, 9):
        blurred_images.append(
            tvf.gaussian_blur(target_images, kernel)
        )
    wfd_list = []
    for blur_image in blurred_images:
        kwd = compute_klwd(target_images=target_images,
                           output_images=blur_image)
        wfd_list.append(kwd)
    assert all(a < b for a, b in pairwise(wfd_list))


@pytest.mark.slow
def test_gaussian_noise():
    target_images = get_images(256)
    noised_images = []
    for noise_ratio in (0.25, 0.5, 0.75, 1):
        noised_images.append(
            noise_ratio * th.randn_like(target_images) + (1 - noise_ratio) * target_images
        )
    wfd_list = []
    for noise_image in noised_images:
        kwd = compute_klwd(target_images=target_images,
                           output_images=noise_image)
        wfd_list.append(kwd)
    assert all(a < b for a, b in pairwise(wfd_list))
