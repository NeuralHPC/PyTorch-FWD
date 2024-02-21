"""Test Wavelet packet Frechet distance."""
import pytest
import torch
import pytest
import numpy as np
import numpy.matlib

from itertools import pairwise
from src.freq_math import wavelet_packet_frechet_distance
from sklearn.datasets import load_sample_images
from torchvision import transforms
from copy import deepcopy

@pytest.mark.slow
def get_images(img_size: int = 64) -> torch.Tensor:
    """Generate images of given size.

    Args:
        img_size (int): Integer specifying the desired image size.

    Returns:
        torch.Tensor: Tensor containing images of shape [batchsize, channels, height, width].

    """
    dataset = load_sample_images()
    tower = torch.Tensor(dataset.images[0])
    flower = torch.Tensor(dataset.images[1])
    images = torch.stack([tower, tower, tower, tower, flower, flower, flower, flower], axis=0)
    images = images.type(torch.FloatTensor) / 255.0
    images = torch.permute(images, [0, 3, 1, 2])
    images = transforms.functional.resize(images, (img_size, img_size))
    assert images.shape == (8, 3, img_size, img_size)
    return images


@pytest.mark.slow
def test_same_input():
    target_images = get_images()
    output_images = deepcopy(target_images)
    distance = wavelet_packet_frechet_distance(output=output_images,
                                               target=target_images,
                                               level=2,
                                               wavelet="sym5")
    assert np.allclose(distance, 0.0, atol=1e-3)


@pytest.mark.slow
def test_shuffle_input():
    target_images = get_images()
    # Shuffle the output images
    output_images = deepcopy(target_images)
    permutation = torch.randperm(len(target_images))
    shuffled_images = output_images[permutation, :, :, :]
    assert not torch.allclose(shuffled_images, output_images)
    shuffled_distance = wavelet_packet_frechet_distance(output=shuffled_images,
                                                        target=target_images,
                                                        level=2,
                                                        wavelet="sym5")
    unshuffled_distance = wavelet_packet_frechet_distance(output=output_images,
                                                          target=target_images,
                                                          level=2,
                                                          wavelet="sym5")
    assert np.allclose(shuffled_distance, unshuffled_distance)

@pytest.mark.slow
@pytest.mark.parametrize("wavelet", ["sym5", "db5", "Haar"])
@pytest.mark.parametrize("level", [2, 3, 4])
def test_various_wavelets(wavelet, level):
    target_images = get_images().type(torch.float64)
    output_images = deepcopy(target_images)
    distance = wavelet_packet_frechet_distance(output=output_images,
                                               target=target_images,
                                               level=level,
                                               wavelet=wavelet)
    assert np.allclose(distance, 0.0, atol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("img_size_level", [(32, 1), (64, 2), (128, 3)]) #  (256, 4)
def test_various_image_sizes(img_size_level):
    size, level = img_size_level
    target_images = get_images(size)
    output_images = deepcopy(target_images)
    assert output_images.shape == (8, 3, size, size)
    distance = wavelet_packet_frechet_distance(output=output_images,
                                               target=target_images,
                                               level=level,
                                               wavelet="sym5")
    assert np.allclose(distance, 0.0, atol=1e-3)


def test_checkerboard():
    import scipy.ndimage

    range_max = 100 
    # tile_size = 10 # determines tile size
    images = []

    for tile_size in [2, 6, 10]:
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
            input_ = numpy.fft.fft2(grid)
            shift = np.random.uniform(-1, 1)*tile_size//2
            image = scipy.ndimage.fourier_shift(input_, shift=shift)
            image = numpy.fft.ifft2(image)
            img_list.append(image.real)
        # B, H, W
        img_array = np.stack(img_list, axis=0)
        # B, C, H, W
        img_array = np.expand_dims(img_array, 1)
        # img_array = np.stack([img_array]*3, 1)
        images.append(img_array)

    # a list for the wavelet frecet distances.
    wfd_list = []
    reference = images[0]
    for cimgs in images[1:]:
        wfd_list.append(wavelet_packet_frechet_distance(output=torch.from_numpy(cimgs),
                                                        target=torch.from_numpy(reference),
                                                        level=3,
                                                        wavelet="sym5"))

    assert all(a < b for a, b in pairwise(wfd_list))