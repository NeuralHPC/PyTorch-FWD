from itertools import product
from typing import Tuple

import numpy as np
import ptwt
import pywt
import torch
import torch.nn.functional as F

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


def compute_kl_divergence(output: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return target * torch.log(((target + eps) / (output + eps)))


def fourier_power_divergence(
    output: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Power spectrum entropy metric as presented in:
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Hernandez_Human_Motion_Prediction_via_Spatio-Temporal_Inpainting_ICCV_2019_paper.pdf

    Args:
        output (torch.Tensor): The network output.
        target (torch.Tensor): The target image.

    Returns:
        (torch.Tensor): A scalar metric.
    """
    assert output.shape == target.shape, "Sampled and reference images should have same shape."
    
    output_fft = torch.abs(torch.fft.fft2(output)) ** 2
    target_fft = torch.abs(torch.fft.fft2(target)) ** 2
    
    b, c, _, _  = output_fft.shape
    
    output_power = torch.reshape(output_fft, (c, -1))
    target_power = torch.reshape(target_fft, (c, -1))
    output_power = output_power / torch.sum(output_power, dim=-1, keepdim=True)
    target_power = target_power / torch.sum(target_power, dim=-1, keepdim=True)


    kld_AB = compute_kl_divergence(output_power, target_power)
    kld_BA = compute_kl_divergence(target_power, output_power)
    
    kld_AB = torch.sum(kld_AB, dim=-1)
    kld_BA = torch.sum(kld_BA, dim=-1)
    return torch.mean(kld_AB), torch.mean(kld_BA)


def wavelet_packet_power_divergence(
    output: torch.Tensor, target: torch.Tensor, level: int = 3, wavelet: str = 'sym5'
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
        level  (int): Wavelet level to use. Defaults to 3
        wavelet(str): Type of wavelet to use. Defaults to db5 

    Returns:
        torch.Tensor: Wavelet power divergence metric
    """
    assert output.shape == target.shape, "Sampled and reference images should have same shape."
    print(f"Using wavelet: {wavelet} with level: {level}")

    output_packets = forward_wavelet_packet_transform(output, max_level=level, wavelet=wavelet)
    target_packets = forward_wavelet_packet_transform(target, max_level=level, wavelet=wavelet)

    output_energy = torch.abs(output_packets) ** 2
    target_energy = torch.abs(target_packets) ** 2

    b, p, c, _, _ = output_packets.shape
    output_energy = output_energy.reshape((p, c, -1))
    target_energy = target_energy.reshape((p, c, -1))
    
    output_power = output_energy / torch.sum(output_energy, dim=-1, keepdim=True)
    target_power = target_energy / torch.sum(target_energy, dim=-1, keepdim=True)
    del output_energy
    del target_energy
    del output_packets
    del target_packets

    kld_AB = compute_kl_divergence(output_power, target_power)
    kld_BA = compute_kl_divergence(target_power, output_power)
    del output_power
    del target_power
    
    kld_AB = torch.sum(kld_AB, dim=-1)
    kld_BA = torch.sum(kld_BA, dim=-1)
    return torch.mean(kld_AB), torch.mean(kld_BA)


def compute_frechet_distance(mu1: np.ndarray, mu2: np.ndarray, sigma1: np.ndarray, sigma2: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Numpy implementation of the Frechet Distance.
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


def wavelet_packet_frechet_distance(
        output: torch.Tensor, target: torch.Tensor, level: int = 3, wavelet: str = 'sym5'
) -> np.ndarray:
    """ Compute the frechet packet distance.

    Args:
        output (torch.Tensor): The network output
        target (torch.Tensor): The target image
        level (int, optional): Wavelet level to use. Defaults to 3
        wavelet (str, optional): Type of wavelet to use. Defaults to 'sym5'

    Returns:
        np.ndarray: Frechet packet distance
    """
    assert output.shape == target.shape, "Sampled and reference images should have same shape."
    print(f"Using wavelet: {wavelet} with level: {level}")

    output_packets = forward_wavelet_packet_transform(output, max_level=level, wavelet=wavelet)
    target_packets = forward_wavelet_packet_transform(target, max_level=level, wavelet=wavelet)

    b, p, c, h, w = output_packets.shape
    output_energy = np.reshape(np.absolute(output_packets.numpy()) ** 2, (b*p*c, h*w))
    del output_packets
    target_energy = np.reshape(np.absolute(target_packets.numpy()) ** 2, (b*p*c, h*w))
    del target_packets

    mu1 = np.mean(output_energy, axis=0)
    sigma1 = np.cov(output_energy, rowvar=False)

    mu2 = np.mean(target_energy, axis=0)
    sigma2 = np.cov(target_energy, rowvar=False)

    fwd = compute_frechet_distance(
        mu1=mu1, mu2=mu2,
        sigma1=sigma1, sigma2=sigma2
    )
    return fwd


def fourier_frechet_distance(
        output: torch.Tensor, target: torch.Tensor
) -> np.ndarray:
    """ Compute frechet frequency distance.

    Args:
        output (torch.Tensor): The network output
        target (torch.Tensor): The target image

    Returns:
        np.ndarray: Frechet frequency distance
    """
    assert output.shape == target.shape, "Sampled and reference images should have same shape."

    output_fft = torch.abs(torch.fft.fft2(output)) ** 2
    target_fft = torch.abs(torch.fft.fft2(target)) ** 2
    
    b, c, h, w = output_fft.shape
    output_fft = np.reshape(output_fft.numpy(), (b*c, h*w))
    target_fft = np.reshape(target_fft.numpy(), (b*c, h*w))

    mu1 = np.mean(output_fft, axis=0)
    sigma1 = np.cov(output_fft, rowvar=False)

    mu2 = np.mean(target_fft, axis=0)
    sigma2 = np.cov(target_fft, rowvar=False)

    ffd = compute_frechet_distance(
        mu1=mu1, mu2=mu2,
        sigma1=sigma1, sigma2=sigma2
    )
    return ffd