"""Test Wavelet packet Frechet distance."""

import torch
import pytest

from src.freq_math import wavelet_packet_frechet_distance
from sklearn.datasets import load_sample_images
from copy import deepcopy


def get_images() -> torch.Tensor:
    """
    Generate sample images for testing.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, channels, height, width).
    """
    dataset = load_sample_images()
    tower = torch.Tensor(dataset.images[0])
    flower = torch.Tensor(dataset.images[1])
    images = torch.stack([tower, tower, flower, flower], axis=0)
    images = images.type(torch.FloatTensor) / 255.0
    images = torch.permute(images, [0, 3, 1, 2])
    assert images.shape == (4, 3, 427, 640)
    return images


def test_same_input():
    target_images = get_images()
    output_images = deepcopy(target_images)
    distance = wavelet_packet_frechet_distance(output=output_images,
                                               target=target_images,
                                               level=3,
                                               wavelet="sym5")
    assert torch.allclose(distance, 0.0)


def test_shuffle_input():
    target_images = get_images()
    # Shuffle the output images
    output_images = deepcopy(target_images)
    permutation = torch.randperm(len(target_images))
    output_images = output_images[permutation, :, :, :]
    distance = wavelet_packet_frechet_distance(output=output_images,
                                               target=target_images,
                                               level=3,
                                               wavelet="sym5")
    assert torch.allclose(distance, 0.0)


@pytest.mark.parametrize("wavelet", ["sym5", "db5", "Haar", "db2", "sym4", "db3", "db4"])
@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_various_wavelet(wavelet, level):
    target_images = get_images()
    output_images = deepcopy(target_images)
    distance = wavelet_packet_frechet_distance(output=output_images,
                                               target=target_images,
                                               level=level,
                                               wavelet=wavelet)
    assert torch.allclose(distance, 0.0)