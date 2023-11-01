import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from glob import glob
import os
import ptwt
import pywt
from src.freq_math import forward_wavelet_packet_transform
from itertools import product


original_path = glob('../cifar_data/cifar10_train/*.jpg')
sample_path_wave = glob('./sample_imgs_DDPM_CIFAR10_PACKET_2023-10-31_19-28-42-304774_seed_42/sample_imgs_torch/*.jpg')
sample_path_mse = glob('/home/lveerama/results/metrics_sampled_images/cifar10/DDPM/sample_imgs_torch/*.jpg')
level = 2
wvlt = 'sym4'


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
            packet = packet_array[index]
            row.append(packet)
        image.append(np.concatenate(row, -1))
    return np.concatenate(image, 0)

def get_images(path):
    imgs = []
    for img in path:
        imgs.append(np.array(Image.open(img).convert('RGB')))
    imgs = np.stack(imgs, axis=0)
    return torch.from_numpy(imgs)


def main():
    global original_path, sample_path_wave, level, wvlt, sample_path_mse
    freq_path, natural_path = get_freq_order(level=level)
    # Read and preprocess image
    print('Loading and computing packets for the original dataset')
    original_tensor = get_images(original_path)
    original_tensor = original_tensor.permute(0, 3, 1, 2)
    original_tensor = original_tensor/255.
    original_packets = forward_wavelet_packet_transform(original_tensor, wavelet=wvlt, max_level=level, log_scale=True)
    del original_tensor
    
    print("Loading and computing packets for Wavelet loss trained dataset")
    wave_tensor = get_images(sample_path_wave)
    wave_tensor = wave_tensor.permute(0, 3, 1, 2)
    wave_tensor = wave_tensor/255.
    wave_packets = forward_wavelet_packet_transform(wave_tensor, wavelet=wvlt, max_level=level, log_scale=True)
    del wave_tensor

    print("Loading and computing packets for mse loss trained dataset")
    mse_tensor = get_images(sample_path_mse)
    mse_tensor = mse_tensor.permute(0, 3, 1, 2)
    mse_tensor = mse_tensor/255.
    mse_packets = forward_wavelet_packet_transform(mse_tensor, wavelet=wvlt, max_level=level, log_scale=True)
    del mse_tensor

    # Generate packets and mean packets
    mean_packets_original = torch.mean(original_packets, dim=(0, 2))
    mean_packets_wave = torch.mean(wave_packets, dim=(0, 2))
    mean_packets_mse = torch.mean(mse_packets, dim=(0, 2))
    # Generate plots - mean packet
    plot_real = generate_frequency_packet_image(mean_packets_original, level)
    plot_mse = generate_frequency_packet_image(mean_packets_mse, level)
    plot_wave = generate_frequency_packet_image(mean_packets_wave, level)
    fig = plt.figure(figsize=(9,3))
    fig.add_subplot(1, 3, 1)
    plt.imshow(plot_real, vmax=1.5, vmin=-7)
    plt.title("real")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar()
    fig.add_subplot(1, 3, 2)
    plt.imshow(plot_mse, vmax=1.5, vmin=-7)
    plt.title("MSE")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar()
    fig.add_subplot(1, 3, 3)
    plt.imshow(plot_wave, vmax=1.5, vmin=-7)
    plt.title("Wavelet")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar()
    plt.savefig(f'./packet_plots/mean_packets_{wvlt}_all.png', dpi=600, bbox_inches='tight')
    # plt.close()
    plt.close()

    # Generate mean packets
    plt.plot(torch.mean(mean_packets_original, (-2, -1)).flatten().numpy(), label='real')
    plt.plot(torch.mean(mean_packets_mse, (-2, -1)).flatten().numpy(), label='mse')
    plt.plot(torch.mean(mean_packets_wave, (-2, -1)).flatten().numpy(), linestyle='dashed', label='packet')
    plt.xlabel('mean packets')
    plt.ylabel('magnitude')
    plt.grid()
    plt.legend()
    plt.savefig(f'./packet_plots/packet_magnitude_{wvlt}_all.png', dpi=600, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
    # img_nms = glob('./packet_plots/packet_*.png')
    # nms = []
    # imgs = []
    # for nm in img_nms:
    #     title_nm = nm.split('_')[-1].split('.')[0]
    #     nms.append(title_nm)
    #     imgs.append(np.array(Image.open(nm)))
    # plt.figure(figsize=(15, 10))
    # for idx, img in enumerate(imgs):
    #     plt.subplot(2, 2, idx+1)
    #     plt.imshow(imgs[idx])
    #     plt.title(nms[idx])
    #     plt.axis('off')
    # plt.savefig('./packet_plots/diff_wavelets.png', dpi=600, bbox_inches='tight')
    # plt.show()