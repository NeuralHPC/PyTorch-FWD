from itertools import product

import pywt
import jaxwt as jwt
import jax.numpy as jnp
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


def process_images(tensor: jnp.ndarray, paths: list, max_level: int = 3) -> jnp.ndarray:
    # tensor = jnp.mean(tensor/255., -1)
    packets = jwt.packets.WaveletPacket2D(tensor, pywt.Wavelet("Haar"),
        max_level=max_level)

    packet_list = []
    for node in paths:
        packet_list.append(packets["".join(node)])
    wp_pt = jnp.stack(packet_list, axis=1)
    # return wp_pt
    return jnp.log(jnp.abs(wp_pt) + 1e-12)



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
    return np.concatenate(image, -2)


if __name__ == '__main__':
    import scipy.datasets
    import matplotlib.pyplot as plt
    face = jnp.transpose(scipy.datasets.face(), [2, 0, 1])
    face = face.astype(jnp.float64)/255.

    _, natural_path = get_freq_order(level=3)
    packets = process_images(face, natural_path)
    p_image = generate_frequency_packet_image(packets, 3)
