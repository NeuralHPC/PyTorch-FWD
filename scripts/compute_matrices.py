"""Compute FID and wavelet matrices for real-curated and sampled images."""

from glob import glob
import numpy as np
from typing import List
from PIL import Image
from tqdm import tqdm
from fid.inception import InceptionV3
from fid.fid import calculate_frechet_distance
from src.freq_math import wavelet_packet_power_divergence
import torch
from torchvision import transforms


original_image_path = None
sampled_image_path = None


def compute_fid(img_tensor: torch.Tensor, imgr_tensor: torch.Tensor) -> float:
    """Compute FID.

    Args:
        img_tensor (torch.Tensor): Reference image.
        imgr_tensor (torch.Tensor): Sampled image.

    Returns:
        float: FID between two images.
    """
    bidx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([bidx])
    op1 = model(img_tensor)[0]
    op2 = model(imgr_tensor)[0]
    onp1 = op1.squeeze(3).squeeze(2).cpu().numpy()
    onp2 = op2.squeeze(3).squeeze(2).cpu().numpy()
    m1, s1 = np.mean(onp1, axis=0), np.cov(onp1, rowvar=False)
    m2, s2 = np.mean(onp2, axis=0), np.cov(onp2, rowvar=False)
    fid = calculate_frechet_distance(m1, s1, m2, s2)
    return fid


def load_images(path: str) -> List[torch.Tensor]:
    """Image loading.

    Args:
        path (str): Image path.

    Returns:
        List[torch.Tensor]: List of load images as torch tensors.
    """
    files = glob(path)
    imgs =[]
    for file in tqdm(files):
        img = transforms.ToTensor()(Image.open(file).convert('RGB'))
        imgs.append(img.unsqueeze(0))
    return imgs


def compute_matrix(imgs1: torch.Tensor, imgs2: torch.Tensor, method: str='fid') -> np.ndarray:
    """Compute the stated method for all images.

    Args:
        imgs1 (torch.Tensor): Sampled images.
        imgs2 (torch.Tensor): Original images.
        method (str, optional): Method to compute between two images. Defaults to 'fid'.

    Returns:
        np.ndarray: Final matrix.
    """
    fid = []
    for img1 in tqdm(imgs1):
        per_img_fid = []
        for img2  in imgs2:
            value=None
            if method == 'fid':
                value = compute_fid(img1, img2)
            else:
                a, b = wavelet_packet_power_divergence(img1, img2, 4, 'sym5')
                value = 0.5*(a.item() + b.item())
            per_img_fid.append(value)
        fid.append(per_img_fid)
    return np.asarray(fid)


def save_matrix(matrix: np.ndarray, name: str) -> None:
    """Save the matrix in npy format.

    Args:
        matrix (np.ndarray): Matrix to be saved.
        name (str): Name of the file.
    """
    with open(name, 'wb') as fp:
        np.save(fp, matrix)


def main():
    global original_image_path, sampled_image_path
    if original_image_path is None or sampled_image_path is None:
        raise ValueError("Both paths must be specified.")
    original_images = load_images(original_image_path)
    sampled_images = load_images(sampled_image_path)
    fid_matrix = compute_matrix(sampled_images, original_images)
    packet_matrix = compute_matrix(sampled_images, original_images, 'wavelet')
    save_matrix(fid_matrix, 'fid_mat.npy')
    save_matrix(packet_matrix, 'packet_mat.npy')    


if __name__ == '__main__':
    main()
