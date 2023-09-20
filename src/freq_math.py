from itertools import product
from typing import Tuple

import torch
import ptwt
import pywt
import torch.nn.functional as F
import numpy as np


def get_freq_order(level: int):
    """Get the frequency order for a given packet decomposition level.

    Adapted from:
    https://github.com/PyWavelets/pywt/blob/master/pywt/_wavelet_packets.py
    The code elements denote the filter application order. The filters
    are named following the pywt convention as:
    a - LL, low-low coefficients
    h - LH, low-high coefficients
    v - HL, high-low coefficients
    d - HH, high-high coefficients
    """
    wp_natural_path = list(product(["a", "h", "v", "d"], repeat=level))

    def _get_graycode_order(level, x="a", y="d"):
        graycode_order = [x, y]
        for _ in range(level - 1):
            graycode_order = [x + path for path in graycode_order] + [
                y + path for path in graycode_order[::-1]
            ]
        return graycode_order

    def _expand_2d_path(path):
        expanded_paths = {"d": "hh", "h": "hl", "v": "lh", "a": "ll"}
        return (
            "".join([expanded_paths[p][0] for p in path]),
            "".join([expanded_paths[p][1] for p in path]),
        )

    nodes: dict = {}
    for (row_path, col_path), node in [
        (_expand_2d_path(node), node) for node in wp_natural_path
    ]:
        nodes.setdefault(row_path, {})[col_path] = node
    graycode_order = _get_graycode_order(level, x="l", y="h")
    nodes_list: list = [nodes[path] for path in graycode_order if path in nodes]
    wp_frequency_path = []
    for row in nodes_list:
        wp_frequency_path.append([row[path] for path in graycode_order if path in row])
    return wp_frequency_path, wp_natural_path


def fold_channels(input_tensor: torch.Tensor) -> torch.Tensor:
    """Fold a trailing (color-) channel into the batch dimension.

    Args:
        input_tensor (torch.Tensor): An array of shape [B, C, H, W]

    Returns:
        torch.Tensor: The folded [B*C, H, W] image.
    """
    shape = input_tensor.shape
    return torch.reshape(input_tensor, (-1, shape[-2], shape[-1]))


def unfold_channels(
    input_tensor: torch.Tensor, original_shape: Tuple[int, int, int, int]
) -> torch.Tensor:
    """Restore channels from the leading batch-dimension.

    Args:
        array (torch.Tensor): An [B*C, packets, H, W] input array.

    Returns:
        torch.Tensor: Output of shape [B, packets, H, W, C]
    """

    _, packets, _, _ = input_tensor.shape
    b, c, h, w = original_shape
    return torch.reshape(input_tensor, (b, packets, c, h, w))


def forward_wavelet_packet_transform(
    tensor: torch.Tensor,
    wavelet: str = "db3",
    max_level: int = 3,
    log_scale: bool = False,
) -> torch.Tensor:
    tensor = tensor.type(torch.FloatTensor)
    packets = ptwt.WaveletPacket2D(tensor, pywt.Wavelet(wavelet), maxlevel=max_level)
    packet_list = []

    for node in packets.get_natural_order(max_level):
        packet = torch.squeeze(packets[node], dim=1)
        packet_list.append(packet)
    wp_pt_rs = torch.stack(packet_list, axis=1)
    if log_scale:
        wp_pt_rs = torch.log(torch.abs(wp_pt_rs) + 1e-12)

    return wp_pt_rs


def generate_frequency_packet_image(packet_array: np.ndarray, degree: int):
    """Create a ready-to-polt image with frequency-order packages.
       Given a packet array in natural order, creat an image which is
       ready to plot in frequency order.
    Args:
        packet_array (np.ndarray): [packet_no, packet_height, packet_width]
            in natural order.
        degree (int): The degree of the packet decomposition.
    Returns:
        [np.ndarray]: The image of shape [original_height, original_width]
    """
    wp_freq_path, wp_natural_path = get_freq_order(degree)

    image = []
    # go through the rows.
    for row_paths in wp_freq_path:
        row = []
        for row_path in row_paths:
            index = wp_natural_path.index(row_path)
            packet = packet_array[:, index]
            row.append(packet)
        image.append(np.concatenate(row, -1))
    return np.concatenate(image, 2)


def inverse_wavelet_packet_transform(
    packet_tensor: torch.Tensor, wavelet: str, max_level: int
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
                torch.unsqueeze(rec, 1), [batch, channels, height, width]
            )
            rec = torch.squeeze(rec, 1)
            wp_dict[node] = rec
    return rec


def fourier_power_divergence(
    output: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Power spectrum entropy metric as presented in:
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Hernandez_Human_Motion_Prediction_via_Spatio-Temporal_Inpainting_ICCV_2019_paper.pdf

    Args:
        output (torch.Tensor): The network output.
        target (torch.Tenso): The target image.

    Returns:
        (torch.Tensor): A scalar metric.
    """

    radius_no_sqrt = lambda z_comp: torch.real(z_comp) ** 2 + torch.imag(z_comp) ** 2

    output_fft = torch.fft.fft2(output)
    output_power = radius_no_sqrt(output_fft)
    target_fft = torch.fft.fft2(target)
    target_power = radius_no_sqrt(target_fft)
    return torch.mean(F.kl_div(torch.log(output_power), target_power))


def wavelet_packet_power_divergence(
    output: torch.Tensor, target: torch.Tensor, level: int = 3
) -> torch.Tensor:
    """Compute the wavelet packet power divergence.

    Daubechies wavelets are orthogonal, see Ripples in Mathematics page 129:
    ""
    For orthogonal trans-
    forms (such as those in the Daubechies family) the number of extra signal
    coefficients is exactly L - 2, with L being the filter length. See p. 135 for the
    proof.
    ""
    Orthogonal transforms conserve energy according
    Proposition 7.7.1 from Ripples in Mathematics page 80.

    Args:
        output (torch.Tensor): The network output
        target (torch.Tensor): The target image

    Returns:
        torch.Tensor: Wavelet power divergence metric
    """
    output_packets = forward_wavelet_packet_transform(output, max_level=level)
    target_packets = forward_wavelet_packet_transform(target, max_level=level)

    output_energy = output_packets**2
    target_energy = target_packets**2
    return torch.mean(F.kl_div(torch.log(output_energy), target_energy))
