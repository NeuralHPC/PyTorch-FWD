from itertools import product
from typing import Tuple
from functools import partial

import numpy as np
import ptwt
import pywt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from scipy import linalg


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


def forward_wavelet_packet_transform(
    tensor: torch.Tensor,
    wavelet: str,
    max_level: int,
    log_scale: bool,
) -> torch.Tensor:
    # ideally the output dtype should depend in the input.
    # tensor = tensor.type(torch.FloatTensor)
    packets = ptwt.WaveletPacket2D(tensor, pywt.Wavelet(wavelet), maxlevel=max_level)
    packet_list = [packets[node] for node in packets.get_natural_order(max_level)]

    # for node in packets.get_natural_order(max_level):
        # packet = torch.squeeze(packets[node], dim=1)
        # packet_list.append(packets[node])
    wp_pt_rs = torch.stack(packet_list, axis=1)
    if log_scale:
        wp_pt_rs = torch.log(torch.abs(wp_pt_rs) + 1e-6)

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


# def compute_kl_divergence(output: torch.Tensor, target: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
#     """Compute KL Divergence

#     Args:
#         output (torch.Tensor): Output Images
#         target (torch.Tensor): Target Images
#         eps (float, optional): Epsilon. Defaults to 1e-12.

#     Returns:
#         torch.Tensor: KL Divergence value
#     """
#     # Tried with eps 1e-30 and this improves the precision by a small margin but overall ranking remains the same
#     return target * torch.log((target / (output + eps)) + eps)


# def wavelet_packet_power_divergence(
#     output: torch.Tensor, target: torch.Tensor, level: int, wavelet: str, log_scale: bool) -> torch.Tensor:
#     """Compute the wavelet packet power divergence.

#     Daubechies wavelets are orthogonal, see Ripples in Mathematics page 129:
#     ""
#     For orthogonal trans-
#     forms (such as those in the Daubechies family) the number of extra signal
#     coefficients is exactly L - 2, with L being the filter length. See p. 135 for the
#     proof.
#     ""
#     Orthogonal transforms conserve energy according
#     Proposition 7.7.1 from Ripples in Mathematics page 80.

#     Args:
#         output (torch.Tensor): The network output
#         target (torch.Tensor): The target image
#         level  (int): Wavelet level to use. Defaults to 3
#         wavelet(str): Type of wavelet to use. Defaults to sym5 

#     Returns:
#         torch.Tensor: Wavelet power divergence metric
#     """
#     assert output.shape == target.shape, "Sampled and reference images should have same shape."
    
#     output_packets = batched_packet_transform(output, max_level=level, wavelet=wavelet, log_scale=log_scale, batch_size=2500)
#     target_packets = batched_packet_transform(target, max_level=level, wavelet=wavelet, log_scale=log_scale, batch_size=2500)

#     B, P, C, H, W = output_packets.shape
#     output_packets = torch.reshape(output_packets, (B, P, C, H*W))
#     target_packets = torch.reshape(target_packets, (B, P, C, H*W))

#     B, P, C, Px = output_packets.shape
#     assert output_packets.shape == target_packets.shape, "Reshape shapes are not same."

#     p_tar_hists = []
#     p_out_hists = []
#     for pindex in range(P):
#         c_tar_hists = []
#         c_out_hists = []
#         for cindex in range(C):
#             op_pack = output_packets[:, pindex, cindex, :].flatten()
#             tg_pack = target_packets[:, pindex, cindex, :].flatten()
#             max_val = torch.max(torch.max(torch.abs(op_pack)), torch.max(torch.abs(tg_pack)))
#             max_val = 1e-12 if max_val == 0 else max_val
#             op_pack = op_pack / max_val
#             tg_pack = tg_pack / max_val
#             output_hist, _ = torch.histogram(op_pack, bins=int((B*Px)**0.5), range=(-1, 1))
#             target_hist, _ = torch.histogram(tg_pack, bins=int((B*Px)**0.5), range=(-1, 1))
#             c_tar_hists.append(target_hist)
#             c_out_hists.append(output_hist)
#         p_tar_hists.append(torch.stack(c_tar_hists))
#         p_out_hists.append(torch.stack(c_out_hists))
            
#     output_hist = torch.stack(p_out_hists)
#     target_hist = torch.stack(p_tar_hists)

#     # output_hist = torch.nn.functional.softmax(output_hist)
#     # target_hist = torch.nn.functional.softmax(target_hist)
#     output_hist = output_hist / (torch.sum(output_hist, dim=-1, keepdim=True) + 1e-12)
#     target_hist = target_hist / (torch.sum(target_hist, dim=-1, keepdim=True) + 1e-12)

#     kld_ab = compute_kl_divergence(output_hist, target_hist)
#     kld_ba = compute_kl_divergence(target_hist, output_hist)
#     kld = 0.5 * (kld_ab + kld_ba)
#     return torch.mean(kld).item() # Average kldivergence across packets and channels


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Implementation from https://github.com/bioinf-jku/TTUR/blob/master/fid.py.

    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)
