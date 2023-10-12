
import scipy
import torch

from src.freq_math import (
    generate_frequency_packet_image,
    forward_wavelet_packet_transform,
    inverse_wavelet_packet_transform,
)


def test_loss():
    import scipy.datasets
    import matplotlib.pyplot as plt

    face = torch.Tensor(scipy.datasets.face())
    face = torch.stack([face, face, face, face], axis=0)
    face = face.type(torch.FloatTensor) / 255.0
    b, h, w, c = face.shape
    face = torch.reshape(face, (b, c, h, w))

    packets = forward_wavelet_packet_transform(face, wavelet="Haar", max_level=3)
    p_image = generate_frequency_packet_image(packets, 3)
    assert p_image.shape == (4, 3, 768, 1024)


def test_inverse_wp():
    face = torch.Tensor(scipy.datasets.face())
    face = torch.stack([face, face, face, face], axis=0)
    face = face.type(torch.FloatTensor) / 255.0
    b, h, w, c = face.shape
    face = torch.reshape(face, (b, c, h, w))

    packets = forward_wavelet_packet_transform(face, max_level=3, wavelet="db3")

    reconstruction = inverse_wavelet_packet_transform(
        packets, max_level=3, wavelet="db3"
    )
    assert torch.max(torch.abs(reconstruction[:, :, :768, :1024] - face)) < 1e-5
