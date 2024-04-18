"""Compute Kullbeck-Liebler Wavelet Divergence."""

import os
import pathlib
from typing import List

import torch as th
import torchvision.transforms as tv
from tqdm import tqdm

from src.freq_math import compute_kl_divergence, forward_wavelet_packet_transform
from src.utils import ImagePathDataset, _parse_args

th.set_default_dtype(th.float64)

IMAGE_EXTS = {"jpg", "jpeg", "png"}
NUM_PROCESSES = None


def get_dataloader(path: str, batch_size: int) -> th.utils.data.DataLoader:
    """Wrap dataloader.

    Args:
        path (str): Image path.
        batch_size (int): Batch size.

    Returns:
        th.utils.data.DataLoader: DataLoader.
    """
    posfix_path = pathlib.Path(path)
    img_names = sorted(
        [name for ext in IMAGE_EXTS for name in posfix_path.glob(f"*.{ext}")]
    )
    dataloader = th.utils.data.DataLoader(
        ImagePathDataset(img_names, transforms=tv.ToTensor()),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_PROCESSES,
    )
    return dataloader


def compute_packets(
    dataloader: th.utils.data.DataLoader, wavelet: str, max_level: int, log_scale: bool
) -> th.Tensor:
    """Compute packets for each batch.

    Args:
        dataloader (th.utils.data.DataLoader): dataloader.
        wavelet (str): Choice of wavelet.
        max_level (int): Decomposition level.
        log_scale (bool): Boolean log scale.

    Returns:
        th.Tensor: Packet tensor.
    """
    packets = []
    device = th.device("cuda:0") if th.cuda.is_available() else th.device("cpu")
    for img_batch in tqdm(dataloader):
        if isinstance(img_batch, list):
            img_batch = img_batch[0]
        img_batch = img_batch.to(device)
        packets.append(
            forward_wavelet_packet_transform(
                img_batch, wavelet, max_level, log_scale
            ).cpu()
        )
    packet_tensor = th.cat(packets, dim=0)
    return packet_tensor


# @th.compile(mode="max-autotune")
def wavelet_power_divergence(packets_0: th.Tensor, packets_1: th.Tensor) -> float:
    """Compute Wavelet Power Divergence.

    Args:
        packets_0 (th.Tensor): Original packets.
        packets_1 (th.Tensor): Target packets.

    Returns:
        float: Wavelet power divergence.
    """
    assert packets_0.shape == packets_1.shape, "Packets are of not same shape"
    Bs, P, C, H, W = packets_0.shape
    packets_1 = th.reshape(packets_1, (Bs, P, C, H * W))
    packets_0 = th.reshape(packets_0, (Bs, P, C, H * W))

    def compute_hists(x):
        return th.histogram(x, bins=int(2*(Bs * H * W) ** (1/3)), range=(-1, 1), density=True) # Use rice rule for number of bins

    def normalize_fn(x):
        return x / (th.sum(x, dim=-1, keepdim=True) + 1e-12)

    p_1_hists, p_0_hists = [], []
    for p_ind in tqdm(range(P)):
        c_1_hists, c_0_hists = [], []
        for c_ind in range(C):
            pack_0 = packets_0[:, p_ind, c_ind, :].flatten()
            pack_1 = packets_1[:, p_ind, c_ind, :].flatten()
            max_val = th.max(th.max(th.abs(pack_0)), th.max(th.abs(pack_1)))
            max_val = th.tensor(1e-12) if max_val == 0 else max_val
            pack_0 = pack_0 / max_val
            pack_1 = pack_1 / max_val
            hist_0, hist_1 = compute_hists(pack_0)[0], compute_hists(pack_1)[0]
            c_0_hists.append(hist_0)
            c_1_hists.append(hist_1)
        p_0_hists.append(th.stack(c_0_hists))
        p_1_hists.append(th.stack(c_1_hists))

    # p0_hist = normalize_fn(th.stack(p_0_hists))
    # p1_hist = normalize_fn(th.stack(p_1_hists))
    p0_hist = th.stack(p_0_hists)
    p1_hist = th.stack(p_1_hists)

    packet_kld = []
    for idx in tqdm(range(len(p0_hist))):
        kld_ab = compute_kl_divergence(p0_hist[idx, :, :], p1_hist[idx, :, :])
        kld_ba = compute_kl_divergence(p1_hist[idx, :, :], p0_hist[idx, :, :])
        kld = 0.5 * (kld_ab + kld_ba)
        packet_kld.append(float(th.mean(kld).item()))
    return sum(packet_kld)/len(packet_kld)  # Avg KLD over packets


def compute_klwd(
    paths: List[str], wavelet: str, max_level: int, log_scale: bool, batch_size: int
) -> float:
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
        float: KL Wavelet Divergence.
    """
    for path in paths:
        if not os.path.exists(path):
            raise RuntimeError(f"Invalid path: {path}")

    print(f"Computing packets of given path: {paths[0]}")
    packets_0 = compute_packets(
        get_dataloader(paths[0], batch_size), wavelet, max_level, log_scale
    )

    print(f"Computing packets of given path: {paths[1]}")
    packets_1 = compute_packets(
        get_dataloader(paths[1], batch_size), wavelet, max_level, log_scale
    )
    print("Computing KLWD...(This might take some time)")
    return wavelet_power_divergence(packets_0, packets_1)


def main():
    """Compute Kullbeck-Liebler Wavelet Divergence."""
    global NUM_PROCESSES, IMAGE_EXTS

    th.manual_seed(0)
    th.use_deterministic_algorithms(True)
    args = _parse_args()
    print(args)
    if args.num_processes is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cpus = os.cpu_count()
        NUM_PROCESSES = min(num_cpus, 16) if num_cpus is not None else 0
    else:
        NUM_PROCESSES = args.num_processes
    print(f"#workers: {NUM_PROCESSES}")

    klwd = compute_klwd(
        args.path, args.wavelet, args.max_level, args.log_scale, args.batch_size
    )
    print(f"KLWD: {klwd}")


if __name__ == "__main__":
    main()
