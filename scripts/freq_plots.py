import pickle
from concurrent.futures import ThreadPoolExecutor as Pool
from functools import partial
from glob import glob
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import ptwt
import pywt
import tikzplotlib as tikz
import torch
from PIL import Image
from tqdm import tqdm

original_path = glob("../cifar_data/cifar10_train/*.jpg")
sample_path_wave = glob(
    "./sample_imgs_DDPM_CIFAR10_PACKET_2023-10-31_19-28-42-304774_seed_42/sample_imgs_torch/*.jpg"
)
sample_path_mse = glob(
    "/home/lveerama/results/metrics_sampled_images/cifar10/DDPM/sample_imgs_torch/*.jpg"
)
level = 2
wvlt = "Haar"


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


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


def get_image(image_path):
    try:
        return np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:
        print(f"error: {e}, path: {image_path}")
        return None


def get_images(paths):
    imgs = []
    with Pool(10) as p:
        imgs = p.map(get_image, paths)
    # imgs = list(map(get_image, path))
    imgs = np.stack(list(filter(lambda i: i is not None, imgs)))
    return torch.from_numpy(imgs)


def process(
    tensor: torch.Tensor, paths: list, wavelet: str, level: int, log_scale: bool = True
) -> torch.Tensor:
    packets = ptwt.WaveletPacket2D(tensor, pywt.Wavelet(wavelet), maxlevel=level)
    packet_list = []
    for node in paths:
        packet = torch.squeeze(packets["".join(node)], dim=1)
        packet_list.append(packet)
    wp_pt = torch.stack(packet_list, dim=1)
    if log_scale:
        return torch.log(torch.abs(wp_pt) + 1e-12)
    return wp_pt


def compute_power_spectrum(input_tensor):
    square = input_tensor**2
    return square / np.sum(square**2, axis=(1, 3, 4), keepdims=True)


def plot_freq_celebA():
    to_plot = []
    level = 4
    wavelet = "sym5"
    nested_freq_path, natural_path = get_freq_order(level=level)
    freq_path = []
    for freq in nested_freq_path:
        freq_path.extend(freq)

    original_images = glob(
        "/p/scratch/holistic-vid-westai/veeramacheneni2_scratch/CelebAMask-HQ/data256x256/*.jpg"
    )
    ddpm_images = glob(
        "/p/scratch/holistic-vid-westai/wolter1_scratch/DDPM/sample_imgs_torch/*.jpg"
    )
    ddim_images = glob(
        "/p/scratch/holistic-vid-westai/wolter1_scratch/DDIM/sample_imgs_torch/*.jpg"
    )
    styleswin_images = glob(
        "/p/scratch/holistic-vid-westai/wolter1_scratch/StyleSwin/celeba_256/samples/eval_0/*.png"
    )
    wavediff = glob(
        "/p/scratch/holistic-vid-westai/wolter1_scratch/WaveDiff/celeba_256/*.jpg"
    )
    ddgan = glob(
        "/p/scratch/holistic-vid-westai/wolter1_scratch/DDGAN/celeba_256/*.jpg"
    )
    stylegan2 = glob(
        "/p/scratch/holistic-vid-westai/wolter1_scratch/StyleGAN2/samples_celeba/*.png"
    )

    def compute_packets(file_list, batch_size=1000, wavelet=wavelet):
        my_process = partial(
            process, paths=natural_path, wavelet=wavelet, level=level, log_scale=False
        )

        image_tensor = get_images(file_list).permute(0, 3, 1, 2) / 255.0
        print(f"image_tensor shape {image_tensor.shape}")
        packets = []
        tensor_list = torch.split(image_tensor, batch_size)
        for tensor in tqdm(tensor_list):
            packets_el = my_process(tensor.cuda())
            packets.append(packets_el.cpu())
        packets = torch.cat(packets, 0)
        return packets

    original_packets = compute_packets(original_images)
    packet_image = np.log(
        generate_frequency_packet_image(
            np.mean(compute_power_spectrum(original_packets.numpy()), axis=(0, 2)), 4
        )
        + 1e-24
    )
    plt.imshow(packet_image)
    plt.savefig(f"orig_packets_{wavelet}.pdf")
    plt.clf()
    mean_power_celeba = np.mean(
        compute_power_spectrum(original_packets.numpy()), axis=(0, 2, 3, 4)
    )
    print("gt packets done.")

    styleswin_packets = compute_packets(styleswin_images)
    plt.imshow(
        np.log(
            generate_frequency_packet_image(
                np.mean(compute_power_spectrum(styleswin_packets.numpy()), axis=(0, 2)),
                4,
            )
            + 1e-24
        )
    )
    plt.savefig(f"styleswin_packets_{wavelet}.pdf")
    plt.clf()
    mean_power_styleswin = np.mean(
        compute_power_spectrum(styleswin_packets.numpy()), axis=(0, 2, 3, 4)
    )
    print("styleswin packets done.")

    ddpm_packets = compute_packets(ddpm_images)
    plt.imshow(
        np.log(
            generate_frequency_packet_image(
                np.mean(compute_power_spectrum(ddpm_packets.numpy()), axis=(0, 2)), 4
            )
            + 1e-24
        )
    )
    plt.savefig(f"ddpm_packets_{wavelet}.pdf")
    plt.clf()
    mean_power_ddpm = np.mean(
        compute_power_spectrum(ddpm_packets.numpy()), axis=(0, 2, 3, 4)
    )
    print("ddpm packets done.")

    ddim_packets = compute_packets(ddim_images)
    plt.imshow(
        np.log(
            generate_frequency_packet_image(
                np.mean(compute_power_spectrum(ddim_packets.numpy()), axis=(0, 2)), 4
            )
            + 1e-24
        )
    )
    plt.savefig(f"ddim_packets_{wavelet}.pdf")
    plt.clf()
    mean_power_ddim = np.mean(
        compute_power_spectrum(ddim_packets.numpy()), axis=(0, 2, 3, 4)
    )
    print("ddim packets done.")
    del ddim_packets

    wave_diff_packets = compute_packets(wavediff)
    plt.imshow(
        np.log(
            generate_frequency_packet_image(
                np.mean(compute_power_spectrum(wave_diff_packets.numpy()), axis=(0, 2)),
                4,
            )
            + 1e-24
        )
    )
    plt.savefig(f"wave_diff_packets_{wavelet}.pdf")
    plt.clf()
    mean_power_wavediff = np.mean(
        compute_power_spectrum(wave_diff_packets.numpy()), axis=(0, 2, 3, 4)
    )
    del wave_diff_packets
    print("wave_diff_packets done.")

    ddgan_packets = compute_packets(ddgan)
    plt.imshow(
        np.log(
            generate_frequency_packet_image(
                np.mean(compute_power_spectrum(ddgan_packets.numpy()), axis=(0, 2)), 4
            )
            + 1e-24
        )
    )
    plt.savefig(f"ddgan_packets_{wavelet}.pdf")
    plt.clf()

    mean_power_ddgan = np.mean(
        compute_power_spectrum(ddgan_packets.numpy()), axis=(0, 2, 3, 4)
    )
    del ddgan_packets

    print("ddgan_packets done.")

    stylegan2_packets = compute_packets(stylegan2)
    plt.imshow(
        np.log(
            generate_frequency_packet_image(
                np.mean(compute_power_spectrum(stylegan2_packets.numpy()), axis=(0, 2)),
                4,
            )
            + 1e-24
        )
    )
    plt.savefig(f"stylegan2_packets_{wavelet}.pdf")
    plt.clf()
    mean_power_stylegan2 = np.mean(
        compute_power_spectrum(stylegan2_packets.numpy()), axis=(0, 2, 3, 4)
    )
    del stylegan2_packets
    print("stylegan2_packets done.")

    plt.clf()
    plt.semilogy(mean_power_celeba, label="celebAHQ")
    plt.semilogy(mean_power_ddpm, ".-", label="DDPM")
    plt.semilogy(mean_power_ddim, ".-", label="DDIM")
    plt.semilogy(mean_power_styleswin, ".-", label="Styleswin")
    plt.semilogy(mean_power_wavediff, ".-", label="wavediff")
    plt.semilogy(mean_power_ddgan, ".-", label="ddgan")
    plt.semilogy(mean_power_stylegan2, ".-", label="stylegan2")
    plt.legend()
    plt.savefig(f"packetplot_{wavelet}.pdf")

    diff = lambda mpower: np.abs(mean_power_celeba - mpower)

    plt.clf()
    plt.semilogy(diff(mean_power_ddpm), ".-", label="DDPM")
    plt.semilogy(diff(mean_power_ddim), ".-", label="DDIM")
    plt.semilogy(diff(mean_power_styleswin), ".-", label="Styleswin")
    plt.semilogy(diff(mean_power_wavediff), ".-", label="wavediff")
    plt.semilogy(diff(mean_power_ddgan), ".-", label="ddgan")
    plt.semilogy(diff(mean_power_stylegan2), ".-", label="stylegan2")

    plt.legend()
    plt.savefig(f"diff_packetplot_{wavelet}.pdf")

    breakpoint()

    plt.semilogy(mean_power_celeba, label="celebAHQ")
    plt.semilogy(mean_power_ddpm, ".-", label="DDPM")
    plt.semilogy(mean_power_stylegan2, ".-", label="stylegan2")
    tikz.save(f"packetplot_celebAHQ_DDPM__stylegan2_{wavelet}.tex", standalone=True)
    plt.savefig(f"packetplot_celebAHQ_DDPM__stylegan2_{wavelet}.pdf")

    plt.imshow(
        generate_frequency_packet_image(
            np.abs(compute_power_spectrum(original_packets.numpy()))
            - generate_frequency_packet_image(
                compute_power_spectrum(ddpm_packets.numpy())
            )
        )
    )
    plt.savefig(f"diff_packetimage_{wavelet}".pdf)

    print("done")

    breakpoint()

    # with open('picklepickle.pkl', 'wb') as f:
    #     pickle.dump([original_packets.numpy(),
    #                  ddpm_packets.numpy(),
    #                  ddim_packets.numpy(),
    #                  styleswin_packets.numpy()], f)


def main_tuned():
    global original_path, sample_path_wave, level, wvlt, sample_path_mse
    freq_path, natural_path = get_freq_order(level=level)
    # Read and preprocess image
    print("Loading and computing packets for the original dataset")
    original_tensor = get_images(original_path)
    original_tensor = original_tensor.permute(0, 3, 1, 2)
    original_tensor = original_tensor / 255.0
    original_packets = process(
        original_tensor, freq_path, wavelet=wvlt, level=level, log_scale=False
    )
    del original_tensor

    print("Loading and computing packets for Wavelet loss trained dataset")
    wave_tensor = get_images(sample_path_wave)
    wave_tensor = wave_tensor.permute(0, 3, 1, 2)
    wave_tensor = wave_tensor / 255.0
    wave_packets = process(
        wave_tensor, freq_path, wavelet=wvlt, level=level, log_scale=False
    )
    del wave_tensor

    print("Loading and computing packets for mse loss trained dataset")
    mse_tensor = get_images(sample_path_mse)
    mse_tensor = mse_tensor.permute(0, 3, 1, 2)
    mse_tensor = mse_tensor / 255.0
    mse_packets = process(
        mse_tensor, freq_path, wavelet=wvlt, level=level, log_scale=False
    )
    del mse_tensor

    # original_packets = original_packets / (torch.amax(torch.abs(original_packets), dim=(0, 2, 3, 4))+1e-12)
    # wave_packets = wave_packets / (torch.amax(torch.abs(wave_packets), dim=(0, 2, 3, 4))+1e-12)
    # mean_packets = mean_packets / (torch.amax(torch.abs(mean_packets), dim=(0, 2, 3, 4))+1e-12)
    # Generate packets and mean packets
    mean_packets_original = torch.mean(original_packets, dim=(0, 2))
    mean_packets_wave = torch.mean(wave_packets, dim=(0, 2))
    mean_packets_mse = torch.mean(mse_packets, dim=(0, 2))

    mean_a = mean_packets_original - mean_packets_mse
    mean_b = mean_packets_original - mean_packets_wave
    mean_a = torch.abs(mean_a * mean_a)
    mean_b = torch.abs(mean_b**2)

    # mean_packets_mse = mean_packets_mse / torch.amax(torch.abs(mean_packets_mse), dim=(1, 2), keepdim=True)
    # mean_packets_wave = mean_packets_wave / torch.amax(torch.abs(mean_packets_wave), dim=(1, 2), keepdim=True)
    # mean_packets_original = mean_packets_original / torch.amax(torch.abs(mean_packets_original), dim=(1, 2), keepdim=True)

    # Generate plots - mean packet
    plot_real = generate_frequency_packet_image(mean_packets_original, level)
    plot_mse = generate_frequency_packet_image(mean_packets_mse, level)
    plot_wave = generate_frequency_packet_image(mean_packets_wave, level)
    fig = plt.figure(figsize=(9, 3))
    fig.add_subplot(1, 3, 1)
    plt.imshow(plot_real, vmax=1, vmin=-1)
    plt.title("real")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar()
    fig.add_subplot(1, 3, 2)
    plt.imshow(plot_mse, vmax=1, vmin=-1)
    plt.title("MSE")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar()
    fig.add_subplot(1, 3, 3)
    plt.imshow(plot_wave, vmax=1, vmin=-1)
    plt.title("Wavelet")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar()
    plt.show()
    # # plt.savefig(f'./packet_plots/mean_packets_{wvlt}_all.png', dpi=600, bbox_inches='tight')
    # # plt.close()

    # # # Tikzplot save
    # fig = plt.gcf()
    # tikz.save('./freq_plots/mean_packet_representation.tex', standalone=True)
    # plt.savefig('./freq_plots/mean_packet_representation.pdf', bbox_inches='tight')

    # plt.close()

    # Generate mean packets magnitude plots
    plt.semilogy(torch.mean(mean_a, (-2, -1)).flatten().numpy(), label="real-mse")
    plt.semilogy(torch.mean(mean_b, (-2, -1)).flatten().numpy(), label="real-wave")
    # plt.plot(torch.mean(mean_packets_wave, (-2, -1)).flatten().numpy(), label='packet')
    plt.xlabel("mean packets")
    plt.ylabel("magnitude")
    plt.grid()
    plt.legend()
    # fig = plt.gcf()
    # fig = tikzplotlib_fix_ncols(fig)
    # tikz.save('./freq_plots/mean_packet_magnitude.tex', standalone=True)
    # plt.savefig('./freq_plots/mean_packet_magnitude.pdf', bbox_inches='tight')
    # # plt.savefig(f'./packet_plots/packet_magnitude_{wvlt}_all.png', dpi=600, bbox_inches='tight')
    # plt.close()
    plt.show()


if __name__ == "__main__":
    plot_freq_celebA()
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
