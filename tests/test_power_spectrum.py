from itertools import pairwise

import pytest
import scipy.datasets as datasets
import torch
import numpy as np
from copy import deepcopy

from src.freq_math import wavelet_packet_power_divergence
from tests.test_wavelet_frechet_distance import get_images

@pytest.mark.slow
def test_same_input():
    target_images = get_images(128)
    output_images = deepcopy(target_images)
    kld = wavelet_packet_power_divergence(target_images, output_images, level=4)
    assert np.allclose(kld, 0.)


@pytest.mark.slow
def test_shuffle_input():
    target_images = get_images(128)
    output_images = deepcopy(target_images)
    permutation = torch.randperm((len(target_images)))
    shuffled_images = output_images[permutation, :, :, :]
    assert not torch.allclose(shuffled_images, output_images)
    kld_shuffled = wavelet_packet_power_divergence(target_images, shuffled_images, level=4)
    kld_original = wavelet_packet_power_divergence(target_images, output_images, level=4)
    assert np.allclose(kld_original, kld_shuffled)
    assert np.allclose(kld_original, 0.)
    assert np.allclose(kld_shuffled, 0.)



def test_checkerboard_power_KL():
    import scipy.ndimage

    range_max = 100 
    # tile_size = 10 # determines tile size
    images = []

    for tile_size in [2, 10, 20]:
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
        for _ in range(10):
            input_ = np.fft.fft2(grid)
            shift = np.random.uniform(-1, 1)*tile_size//2
            image = scipy.ndimage.fourier_shift(input_, shift=shift)
            image = np.fft.ifft2(image)
            img_list.append(image.real)
        # B, H, W
        img_array = np.stack(img_list, axis=0)
        # B, C, H, W
        # img_array = np.expand_dims(img_array, 1)
        img_array = np.stack([img_array]*3, 1) # TODO!
        images.append(img_array)

    # a list for the wavelet frecet distances.
    wfd_list = []
    reference = images[0]
    for cimgs in images[1:]:
        wfd_list.append(wavelet_packet_power_divergence(output=torch.from_numpy(cimgs),
                                                        target=torch.from_numpy(reference),
                                                        level=6,
                                                        wavelet="haar"))

    assert all(a < b for a, b in pairwise(wfd_list))

