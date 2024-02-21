from itertools import product
from typing import Tuple
from functools import partial

import numpy as np
import ptwt
import pywt
import torch
import torch.nn.functional as F
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
    # ideally the output dtype should depend in the input.
    # tensor = tensor.type(torch.FloatTensor)
    packets = ptwt.WaveletPacket2D(tensor, pywt.Wavelet(wavelet), maxlevel=max_level)
    packet_list = []

    for node in packets.get_natural_order(max_level):
        packet = torch.squeeze(packets[node], dim=1)
        packet_list.append(packet)
    wp_pt_rs = torch.stack(packet_list, axis=1)
    if log_scale:
        wp_pt_rs = torch.log(torch.abs(wp_pt_rs) + 1e-12)

    return wp_pt_rs


def batched_packet_transform(
        tensor: torch.Tensor,
        wavelet: str = "sym5",
        max_level: int = 4,
        log_scale: bool = False,
        batch_size: int = 2500
) -> torch.Tensor:
    """Compute wavelet packet transform over batches.

    Args:
        tensor (torch.Tensor): Input tensor of shape [BS, CHANNELS, HEIGHT, WIDTH]
        wavelet (str): Choice of wavelet
        max_level (int): Maximum decomposition level
        log_scale (bool): Whether to apply log scale
        batch_size (int): Batch size for tensor split

    Returns:
        torch.Tensor: Tensor containing packets of shape [BS, PACKETS, CHANNELS, HEIGHT, WIDTH]

    """
    assert len(tensor.shape) == 4, "Input tensor for packet transforms must have 4 dimensions"
    batched_tensor = tensor.split(2500, dim=0)
    packets = []
    for image_batch in tqdm(batched_tensor):
        packets.append(forward_wavelet_packet_transform(image_batch, wavelet, max_level, log_scale))
    return torch.cat(packets, dim=0)


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
    """Compute KL Divergence

    Args:
        output (torch.Tensor): Output Images
        target (torch.Tensor): Target Images
        eps (float, optional): Epsilon. Defaults to 1e-12.

    Returns:
        torch.Tensor: KL Divergence value
    """
    # Tried with eps 1e-30 and this improves the precision by a small margin but overall ranking remains the same
    return target * torch.log(((target) / (output + eps)) + eps)


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

    TODO:REMOVE
    """
    assert output.shape == target.shape, "Sampled and reference images should have same shape."
    
    output_fft = torch.abs(torch.fft.fft2(output)) ** 2
    target_fft = torch.abs(torch.fft.fft2(target)) ** 2
    
    b, c, _, _  = output_fft.shape
    
    output_fft = output_fft.swapaxes(0, 1)
    target_fft = target_fft.swapaxes(0, 1)
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
        wavelet(str): Type of wavelet to use. Defaults to sym5 

    Returns:
        torch.Tensor: Wavelet power divergence metric
    """
    assert output.shape == target.shape, "Sampled and reference images should have same shape."
    
    output_packets = batched_packet_transform(output, max_level=level, wavelet=wavelet)
    target_packets = batched_packet_transform(target, max_level=level, wavelet=wavelet)

    B, P, C, H, W = output_packets.shape
    output_packets = torch.reshape(output_packets, (B, P, C, H*W))
    target_packets = torch.reshape(target_packets, (B, P, C, H*W))

    B, P, C, Px = output_packets.shape
    #o utput_packets = torch.reshape(output_packets, (P*C, Px, B))
    # target_packets = torch.reshape(target_packets, (P*C, Px, B))
    assert output_packets.shape == target_packets.shape, "Reshape shapes are not same."

    p_tar_hists = []
    p_out_hists = []
    for pindex in range(P):
        c_tar_hists = []
        c_out_hists = []
        for cindex in range(C):
            pix_out_hists = []
            pix_tar_hists = []
            for pix_index in range(Px):
                output_hist, _ = torch.histogram(output_packets[:, pindex, cindex, pix_index], bins=int(B**0.5))
                target_hist, _ = torch.histogram(target_packets[:, pindex, cindex, pix_index], bins=int(B**0.5))
                pix_out_hists.append(output_hist)
                pix_tar_hists.append(target_hist)
            c_out_hists.append(torch.stack(pix_out_hists))
            c_tar_hists.append(torch.stack(pix_tar_hists))
        p_tar_hists.append(torch.stack(c_tar_hists))
        p_out_hists.append(torch.stack(c_out_hists))
            
    output_hist = torch.stack(p_out_hists)
    target_hist = torch.stack(p_tar_hists)

    output_hist = output_hist.flatten()
    target_hist = target_hist.flatten()

    output_hist = output_hist / torch.sum(output_hist)
    target_hist = target_hist / torch.sum(target_hist)


    kld_ab = compute_kl_divergence(output_hist, target_hist)
    kld_ba = compute_kl_divergence(target_hist, output_hist)
    kld = 0.5 * (kld_ab + kld_ba)
    return kld # Average kldivergence across packets and channels


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Implementation from https://github.com/mseitzer/pytorch-fid.

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


def wavelet_packet_frechet_distance(
        output: torch.Tensor, target: torch.Tensor, level: int = 4, wavelet: str = "sym5"
) -> float:
    """Compute wavelet packet Frechet distance.

    Args:
        output (torch.Tensor): Generated images of shape [B, C, H, W].
        target (torch.Tensor): Ground truth images. Same shape as output.
        level (int, optional): Wavelet decomposition level. Defaults to 4.
        wavelet (str, optional): Wavelet type to use. Defaults to "sym5".

    Returns:
        float: Wavelet packet Frechet distance.
    """
    assert output.shape == target.shape, "Output and target must be of same shape."
    # Compute wavelet packet transform.
    output_packets = batched_packet_transform(tensor=output,
                                              wavelet=wavelet,
                                              max_level=level)
    target_packets = batched_packet_transform(tensor=target,
                                              wavelet=wavelet,
                                              max_level=level)
    assert output_packets.shape == target_packets.shape, "Output & target packets are not of same shape."

    # Permute patches and batch dimensions
    output_packets = torch.permute(output_packets, (1, 0, 2, 3, 4))
    target_packets = torch.permute(target_packets, (1, 0, 2, 3, 4))
    PACKETS, BATCH, CHANNELS, HEIGHT, WIDTH = output_packets.shape

    # Flatten each packet in batch into vector
    output_reshaped = torch.reshape(output_packets, (PACKETS, BATCH, CHANNELS*HEIGHT*WIDTH))
    target_reshaped = torch.reshape(target_packets, (PACKETS, BATCH, CHANNELS*HEIGHT*WIDTH))

    output_array = output_reshaped.detach().cpu().numpy()
    target_array = target_reshaped.detach().cpu().numpy()

    output_means = [np.mean(output_array[packet_no, :, :], axis=0) for packet_no in range(PACKETS)]
    target_means = [np.mean(target_array[packet_no, :, :], axis=0) for packet_no in range(PACKETS)]

    output_covs = [np.cov(output_array[packet_no, :, :], rowvar=False) for packet_no in range(PACKETS)]
    target_covs = [np.cov(target_array[packet_no, :, :], rowvar=False) for packet_no in range(PACKETS)]

    print("Computing per packet FID...")
    frechet_distances = []
    for packet_no in tqdm(range(PACKETS)):
        frechet_distance = calculate_frechet_distance(mu1=output_means[packet_no],
                                                      mu2=target_means[packet_no],
                                                      sigma1=output_covs[packet_no],
                                                      sigma2=target_covs[packet_no])
        frechet_distances.append(frechet_distance)
    return np.mean(frechet_distances)
