import scipy.datasets as datasets
import torch
import numpy as np

from src.freq_math import (
    fourier_power_divergence,
    wavelet_packet_power_divergence
)


def test_fourier_divergence():
    face = torch.Tensor(datasets.face())
    face = torch.stack([face, face, face, face], dim=0)
    face = face / 255.
    face = face.permute(0, 3, 1, 2)
    ab, ba = fourier_power_divergence(face, face)
    assert np.allclose(ab.item(), 0., atol=1e-5)
    assert np.allclose(ba.item(), 0., atol=1e-5)


def test_packet_divergence():
    face = torch.Tensor(datasets.face())
    face = torch.stack([face, face, face, face], dim=0)
    face = face / 255.
    face = face.permute(0, 3, 1, 2)
    ab, ba = wavelet_packet_power_divergence(face, face)
    assert np.allclose(ab.item(), 0., atol=1e-5)
    assert np.allclose(ba.item(), 0., atol=1e-5)
