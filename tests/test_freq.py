from typing import Tuple

import pytest
import scipy
import torch as th
import ptwt
import pywt

from itertools import product
from src.freq_math import (
    generate_frequency_packet_image,
    forward_wavelet_packet_transform
)
from .test_wavelet_frechet_distance import get_images


@pytest.mark.slow
def test_loss():
    import scipy.datasets
    import matplotlib.pyplot as plt

    face = th.Tensor(scipy.datasets.face())
    face = th.stack([face, face, face, face], axis=0)
    face = face.type(th.FloatTensor) / 255.0
    b, h, w, c = face.shape
    face = th.reshape(face, (b, c, h, w))

    packets = forward_wavelet_packet_transform(face, wavelet="Haar", max_level=3, log_scale=False)
    p_image = generate_frequency_packet_image(packets, 3)
    assert p_image.shape == (4, 3, 768, 1024)


def test_inverse_wp():
    face = th.Tensor(scipy.datasets.face())
    face = th.stack([face, face, face, face], axis=0)
    face = face.type(th.FloatTensor) / 255.0
    b, h, w, c = face.shape
    face = th.reshape(face, (b, c, h, w))

    packets = forward_wavelet_packet_transform(face, max_level=3, wavelet="db3", log_scale=False)

    reconstruction = inverse_wavelet_packet_transform(
        packets, max_level=3, wavelet="db3"
    )
    assert th.max(th.abs(reconstruction[:, :, :768, :1024] - face)) < 1e-5



def fold_channels(input_tensor: th.Tensor) -> th.Tensor:
    """Fold a trailing (color-) channel into the batch dimension.

    Args:
        input_tensor (th.Tensor): An array of shape [B, C, H, W]

    Returns:
        th.Tensor: The folded [B*C, H, W] image.
    """
    shape = input_tensor.shape
    return th.reshape(input_tensor, (-1, shape[-2], shape[-1]))


def unfold_channels(
    input_tensor: th.Tensor, original_shape: Tuple[int, int, int, int]
) -> th.Tensor:
    """Restore channels from the leading batch-dimension.

    Args:
        array (th.Tensor): An [B*C, packets, H, W] input array.

    Returns:
        th.Tensor: Output of shape [B, packets, H, W, C]
    """

    _, packets, _, _ = input_tensor.shape
    b, c, h, w = original_shape
    return th.reshape(input_tensor, (b, packets, c, h, w))


def inverse_wavelet_packet_transform(
    packet_tensor: th.Tensor, wavelet: str, max_level: int
):
    batch, _, channels, _, _ = packet_tensor.shape

    def get_node_order(level):
        wp_natural_path = list(product(["a", "h", "v", "d"], repeat=level))
        return ["".join(p) for p in wp_natural_path]

    wp_dict = {}
    for pos, path in enumerate(get_node_order(max_level)):
        wp_dict[path] = packet_tensor[:, pos, :, :, :]

    for level in reversed(range(max_level)):
        for node in get_node_order(level):
            data_a = fold_channels(wp_dict[node + "a"])
            data_h = fold_channels(wp_dict[node + "h"])
            data_v = fold_channels(wp_dict[node + "v"])
            data_d = fold_channels(wp_dict[node + "d"])
            rec = ptwt.waverec2(
                [data_a, (data_h, data_v, data_d)], pywt.Wavelet(wavelet)
            )
            height = rec.shape[1]
            width = rec.shape[2]
            rec = unfold_channels(
                th.unsqueeze(rec, 1), [batch, channels, height, width]
            )
            rec = th.squeeze(rec, 1)
            wp_dict[node] = rec
    return rec
