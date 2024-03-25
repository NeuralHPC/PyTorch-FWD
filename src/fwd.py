from freq_math import forward_wavelet_packet_transform, calculate_frechet_distance
from more_itertools import batched
import torch as th
import os
from typing import List, Tuple
from PIL import Image
from torchvision.transforms import functional as TVF
import numpy as np
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
# from multiprocessing import Pool


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch-size", type=int, default=128, help="Batch size for wavelet packet transform.")
parser.add_argument("--num-processes", type=int, default=1, help="Number of multiprocess.")
parser.add_argument("--save-packets", action="store_true", help="Save the packets as npz file.")
parser.add_argument("--wavelet", type=str, default="Haar", help="Choice of wavelet.")
parser.add_argument("--max_level", type=int, default=4, help="wavelet decomposition level")
parser.add_argument("--log_scale", action="store_false", help="Use log scaling for wavelets.")
parser.add_argument("path", type=str, nargs=2, help="Path to the generated images or path to .npz statistics file.")

IMAGE_EXTS = {"jpg", "jpeg", "png"}
NUM_PROCESSES = 1


def read_image(img_name):
    return TVF.pil_to_tensor(Image.open(img_name).convert("RGB"))/255.


def packet_transform(
        imgs: List[str],
        wavelet: str,
        max_level: int,
        log_scale: bool,
        num_batches: int
) -> th.Tensor:
    """Compute wavelet packet transform across batches.

    Args:
        imgs (List[str]): List containing batched images.
        wavelet (str): Choice of wavelet.
        max_level (int): Wavelet decomposition level.
        log_scale (bool): Apply log scale.

    Returns:
        th.Tensor: Packets of shape [BS, P, C, H_n, W_n].
    """
    packets = []
    for img_batch in tqdm(imgs, total=num_batches):
        image = [read_image(nm) for nm in img_batch]
        # with Pool(NUM_PROCESSES) as p:
        #     image = p.map(read_image, img_batch)
        tensor_ = th.stack(image, dim=0)
        packets.append(
            forward_wavelet_packet_transform(
                tensor_,
                wavelet,
                max_level,
                log_scale
            )
        )
    print(len(packets))
    packet_tensor = th.cat(packets, dim=0)
    packet_tensor = th.permute(packet_tensor, (1, 0, 2, 3, 4))
    P, BS, C, H, W = packet_tensor.shape
    return th.reshape(packet_tensor, (P, BS, C*H*W)).numpy()


def compute_statistics(
        img_names: List[str],
        wavelet: str,
        max_level: int,
        log_scale: bool,
        batch_size: int
) -> Tuple[np.ndarray, ...]:
    """Calculate mean and standard deviation across packets.

    Args:
        img_names (List[str]): List of image file names.
        wavelet (str): Choice of wavelet.
        max_level (int): Wavelet decomposition level.
        log_scale (bool): Apply log scale.
        batch_size (int): Batch size for tensor split.

    Returns:
        Tuple[th.Tensor, th.Tensor]: tuple containing mean and std for each packet.
    """
    img_batches = batched(img_names, batch_size)
    num_bacthes = len(img_names)//batch_size
    packets = packet_transform(img_batches, wavelet, max_level, log_scale, num_bacthes)
    mu, sigma = [], []
    for packet_num in range(len(packets)):
        mu.append(np.mean(packets[packet_num, :, :], axis=0))
        sigma.append(np.cov(packets[packet_num, :, :], rowvar=False))
    return mu, sigma


def calculate_path_statistics(path:str, wavelet: str, max_level: int, log_scale: bool, batch_size: int) -> Tuple[np.ndarray, ...]:
    """Compute mean and sigma for given path.

    Args:
        path (str): npz path or image directory.
        wavelet (str): Choice of wavelet.
        max_level (int): Decomposition level.
        log_scale (bool): Apply log scale.
        batch_size (int): Batch size for packet decomposition.

    Raises:
        ValueError: Error if mu and sigma cannot be calculated.

    Returns:
        Tuple[np.ndarray, ...]: Tuple containing mean and sigma for each packet.
    """
    mu, sigma = None, None
    if path.endswith(".npz"):
        with np.load(path) as fp:
            mu = fp["mu"][:]
            sigma = fp["sigma"][:]
    else:
        path = pathlib.Path(path)
        img_names = sorted(
            [
                name
                for ext in IMAGE_EXTS
                for name in path.glob(f"*.{ext}")
            ]
        )
        mu, sigma = compute_statistics(
            img_names=img_names,
            wavelet=wavelet,
            max_level=max_level,
            log_scale=log_scale,
            batch_size=batch_size
        )
    
    if (mu is None) or (sigma is None):
        raise ValueError(f"The file path: {path} is empty/doesn't have statistics.")
    return mu, sigma


def compute_fwd(paths: List[str], wavelet: str, max_level: int, log_scale: bool, batch_size: int) -> float:
    """Compute Frechet Wavelet Distance.

    Args:
        paths (List[str]): List containing path of source and generated images.
        wavelet (str): Choice of wavelet.
        max_level (int): Decomposition level.
        log_scale (bool): Apply log scale.
        batch_size (int): Batch size for packet decomposition.

    Raises:
        RuntimeError: Error if path doesn't exist.

    Returns:
        float: Frechet Wavelet Distance.
    """
    for path in paths:
        if not os.path.exists(path):
            raise RuntimeError(f"Invalid path: {path}")
    
    print(f"Computing stats for path: {paths[0]}")
    mu_1, sigma_1 = calculate_path_statistics(paths[0], wavelet, max_level, log_scale, batch_size)
    print(f"Computing stats for path: {paths[1]}")
    mu_2, sigma_2 = calculate_path_statistics(paths[1], wavelet, max_level, log_scale, batch_size)

    frechet_distances = []
    print("Computing Frechet distances for each packet.")
    for packet_no in tqdm(range(len(mu_1))):
        fd = calculate_frechet_distance(mu1=mu_1[packet_no], mu2=mu_2[packet_no],
                                        sigma1=sigma_1[packet_no], sigma2=sigma_2[packet_no])
        frechet_distances.append(fd)
    return np.mean(frechet_distances)


def save_packets(paths: List[str], wavelet: str, max_level:int, log_scale: bool, batch_size: int) -> None:
    """Save packets.

    Args:
        paths (List[str]): List of paths containing input and output files.
        wavelet (str): Choice of wavelet.
        max_level (int): Decomposition level.
        log_scale (bool): Apply log scale.
        batch_size (int): Batch size for packet decomposition.

    Raises:
        RuntimeError: Error if input path is invalid.
        RuntimeError: Error if the output file already exists.
    """
    if not os.path.exists(paths[0]):
        raise RuntimeError(f"Invalid path: {paths[0]}")
    
    if os.path.exists(paths[1]):
        raise RuntimeError(f"Stats file already exists at the given path: {paths[1]}")
    
    print(f"Computing stats for path: {paths[0]}")
    mu_1, sigma_1 = calculate_path_statistics(paths[0], wavelet, max_level, log_scale, batch_size)
    np.savez_compressed(paths[1], mu=mu_1, sigma=sigma_1)


def main():
    global NUM_PROCESSES, IMAGE_EXTS

    args = parser.parse_args()
    NUM_PROCESSES = args.num_processes
    if args.save_packets:
        save_packets(args.path, args.wavelet, args.max_level, args.log_scale, args.batch_size)
        return
    
    fwd = compute_fwd(args.path, args.wavelet, args.max_level, args.log_scale, args.batch_size)
    print(f"FWD: {fwd}")


if __name__ == '__main__':
    main()
