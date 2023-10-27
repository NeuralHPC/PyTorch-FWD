import scipy.datasets as datasets
import torch

from src.freq_math import (
    fourier_power_divergence,
    wavelet_packet_power_divergence,
    fourier_frechet_distance,
    wavelet_packet_frechet_distance,
)


def test_fourier_divergence():
    face = torch.Tensor(datasets.face())
    face = torch.stack([face, face, face, face], dim=0)
    face = face.type(torch.FloatTensor) / 255.
    face = face.permute(0, 3, 1, 2)
    ab, ba = fourier_power_divergence(face, face)
    assert ab.item() == 0.
    assert ba.item() == 0.


def test_packet_divergence():
    face = torch.Tensor(datasets.face())
    face = torch.stack([face, face, face, face], dim=0)
    face = face.type(torch.FloatTensor) / 255.
    face = face.permute(0, 3, 1, 2)
    ab, ba = wavelet_packet_power_divergence(face, face)
    assert ab.item() == 0.
    assert ba.item() == 0.


def test_fourier_frechet():
    face = torch.Tensor(datasets.face())
    face = torch.stack([face, face, face, face], dim=0)
    face = face.type(torch.FloatTensor) / 255.
    face = face.permute(0, 3, 1, 2)
    ffd = fourier_frechet_distance(face, face)
    assert ffd == 0


def test_packet_frechet():
    face = torch.Tensor(datasets.face())
    face = torch.stack([face, face, face, face], dim=0)
    face = face.type(torch.FloatTensor) / 255.
    face = face.permute(0, 3, 1, 2)
    ffd = wavelet_packet_frechet_distance(face, face)
    assert ffd == 0