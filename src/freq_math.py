from typing import Tuple

from itertools import product

import pywt
import jaxwt as jwt
import jax.numpy as jnp
import numpy as np
import optax

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


def fold_channels(array: jnp.ndarray) -> jnp.ndarray:
    """Fold a trailing (color-) channel into the batch dimension.

    Args:
        array (jnp.ndarray): An array of shape [B, H, W, C]

    Returns:
        jnp.ndarray: The folded [B*C, H, W] image.
    """    
    shape = array.shape
    # fold color channel.
    return jnp.transpose(jnp.reshape(
        jnp.transpose(array, [1, 2, 0, 3]), [shape[1], shape[2], -1]), [-1, 0, 1])

def unfold_channels(array: jnp.ndarray, original_shape: Tuple[int, int, int, int]) -> jnp.ndarray:
    """Restore channels from the leading batch-dimension.

    Args:
        array (jnp.ndarray): An [B*C, packets, H, W] input array. 

    Returns:
        jnp.ndarray: Output of shape [B, H, W, C]
    """
     
    bc_shape = array.shape
    array_rs = jnp.reshape(jnp.transpose(array, [1, 2, 3, 0]),
                           [bc_shape[1], bc_shape[2], bc_shape[3],
                            original_shape[0], original_shape[3]])
    return jnp.transpose(array_rs, [-2, 0, 1, 2, 4])


def forward_wavelet_packet_transform(
        tensor: jnp.ndarray, wavelet: str = "db3", max_level: int = 3,
        log_scale=False) -> jnp.ndarray:
    shape = tensor.shape
    # fold color channel.
    tensor = fold_channels(tensor)
    packets = jwt.packets.WaveletPacket2D(tensor, pywt.Wavelet(wavelet),
        max_level=max_level)

    paths = list(product(["a", "h", "v", "d"], repeat=max_level))
    packet_list = []
    for node in paths:
        packet_list.append(packets["".join(node)])
    wp_pt = jnp.stack(packet_list, axis=1)

    # restore color channel
    wp_pt_rs = unfold_channels(wp_pt, shape)

    if log_scale:
        wp_pt_rs = jnp.log(jnp.abs(wp_pt_rs) + 1e-12)

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
        image.append(np.concatenate(row, -2))
    return np.concatenate(image, -3)


def inverse_wavelet_packet_transform(packet_array: jnp.array, wavelet: str, max_level: int):
    batch = packet_array.shape[0]
    channel = packet_array.shape[-1]

    def get_node_order(level):
        wp_natural_path = list(product(["a", "h", "v", "d"], repeat=level))
        return ["".join(p) for p in wp_natural_path]

    wp_dict = {}
    for pos, path in enumerate(get_node_order(max_level)):
        wp_dict[path] = packet_array[:, pos, :, :, :]

    for level in reversed(range(max_level)):
        for node in get_node_order(level):
            data_a = fold_channels(wp_dict[node + "a"])
            data_h = fold_channels(wp_dict[node + "h"])
            data_v = fold_channels(wp_dict[node + "v"])
            data_d = fold_channels(wp_dict[node + "d"])
            rec = jwt.waverec2([data_a, (data_h, data_v, data_d)], pywt.Wavelet(wavelet))
            height = rec.shape[1]
            width = rec.shape[2]
            rec = unfold_channels(np.expand_dims(rec, 1), [batch, height, width, channel])
            rec = np.squeeze(rec, 1)
            wp_dict[node] = rec
    return rec


def power_divergence(output: jnp.ndarray, target: jnp.ndarray) -> jnp.array:
    """Power spectrum entropy metric as presented in:
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Hernandez_Human_Motion_Prediction_via_Spatio-Temporal_Inpainting_ICCV_2019_paper.pdf

    Args:
        output (jnp.ndarray): The network output.
        target (jnp.ndarray): The target image.

    Returns:
        (jnp.ndarray): A scalar metric.
    """
    output_power = jnp.fft.fft2(output)**2
    target_power = jnp.fft.fft2(target)**2
    return optax.convex_kl_divergence(jnp.log(output_power), target_power)
