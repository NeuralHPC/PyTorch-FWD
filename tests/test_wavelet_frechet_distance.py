"""Test Wavelet packet Frechet distance."""

import torch
import pytest
import numpy as np

from src.freq_math import wavelet_packet_frechet_distance
from sklearn.datasets import load_sample_images
from torchvision import transforms
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
    images = torch.stack([tower, tower, tower, tower, flower, flower, flower, flower], axis=0)
    images = images.type(torch.FloatTensor) / 255.0
    images = torch.permute(images, [0, 3, 1, 2])
    images = transforms.functional.resize(images, (64, 64))
    assert images.shape == (8, 3, 64, 64)
    return images


def test_same_input():
    target_images = get_images()
    output_images = deepcopy(target_images)
    distance = wavelet_packet_frechet_distance(output=output_images,
                                               target=target_images,
                                               level=2,
                                               wavelet="sym5")
    assert np.allclose(distance, 0.0, atol=1e-3)


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


@pytest.mark.parametrize("wavelet", ["sym5", "db5", "Haar"])
@pytest.mark.parametrize("level", [2, 3, 4])
def test_various_wavelet(wavelet, level):
    target_images = get_images()
    output_images = deepcopy(target_images)
    distance = wavelet_packet_frechet_distance(output=output_images,
                                               target=target_images,
                                               level=level,
                                               wavelet=wavelet)
    assert np.allclose(distance, 0.0, atol=1e-3)