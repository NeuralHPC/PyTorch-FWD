import scipy.datasets as datasets
import torch
import numpy as np
from copy import deepcopy

from src.freq_math import (
    fourier_power_divergence,
    wavelet_packet_power_divergence
)
from tests.test_wavelet_frechet_distance import get_images

def test_fourier_divergence():
    face = torch.Tensor(datasets.face())
    face = torch.stack([face, face, face, face], dim=0)
    face = face / 255.
    face = face.permute(0, 3, 1, 2)
    ab, ba = fourier_power_divergence(face, face)
    assert np.allclose(ab.item(), 0., atol=1e-5)
    assert np.allclose(ba.item(), 0., atol=1e-5)


def test_same_input():
    target_images = get_images(256)
    output_images = deepcopy(target_images)
    kld = wavelet_packet_power_divergence(target_images, output_images, level=4)
    assert np.allclose(kld, 0.)


def test_shuffle_input():
    target_images = get_images(256)
    output_images = deepcopy(target_images)
    permutation = torch.randperm((len(target_images)))
    shuffled_images = output_images[permutation, :, :, :]
    assert not torch.allclose(shuffled_images, output_images)
    kld_shuffled = wavelet_packet_power_divergence(target_images, shuffled_images, level=4)
    kld_original = wavelet_packet_power_divergence(target_images, output_images, level=4)
    assert np.allclose(kld_original, kld_shuffled)
    assert np.allclose(kld_original, 0.)
    assert np.allclose(kld_shuffled, 0.)