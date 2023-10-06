"""Compute FID Score based on the sampled and original dataset."""

import torch
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np
import glob, os
import json
import shutil
from torchvision import transforms, datasets
from PIL import Image
from config import cifar10

from src.freq_math import fourier_power_divergence, wavelet_packet_power_divergence


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transforms_=None):
        self.images = glob.glob(os.path.join(data_path, '*.png'))
        self.transforms = transforms_

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = Image.open(img_name).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def tensor_summary_stats(x):
    """Returns summary statistics about x: shape, mean, std, min, max."""
    return f"shape {x.shape}, values in [{x.min():.3f}, {x.max():.3f}] and around {x.mean():.3f} +- {x.std():.3f}"


def get_activations(num_images, dataloader, model, dims, device):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    pred_arr = np.empty((num_images, dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        if isinstance(batch, list):
            batch = batch[0]  # TensorDataset...
        batch = batch.to(device)

        # Map to [0, 255], quantize, then map to [0, 1]
        # batch -= batch.min()
        # batch *= 255 / batch.max()
        # batch = batch.int().float() / 255
        if start_idx == 0:
            print(f"Got batch with stats {tensor_summary_stats(batch)}")

        with torch.no_grad():
            pred = model(batch)
            pred = pred[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics(num_images, dataloader, model, dims, device):
    """Calculation of the statistics used by the FID.
    Params:
    -- images       : (N, C, H, W) float tensor with values in [0, 1]
    -- model       : Instance of inception model
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(num_images, dataloader, model, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_FID(
    orig_dl,
    synthetic_dl,
    device,
    dims,
):
    """Computes FID between reference dataset and synthetic datasets list.
    Paths can either be folders of images, or numpy arrays of samples.
    :param ref_path: path of folder containing reference dataset. Must contain list of images in .jpg or .png format.
    :param synthetic_paths: list of paths of image folders.
    :param device:
    :param num_workers:
    :param batch_size:
    :param dims: dimension used for the Inception network
    :return: a dictionary with keys = the synthetic paths and value = the corresponding FID wrt the reference.
    """
    print(f"Using dims: {dims}")
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    mean_ref, cov_ref = calculate_activation_statistics(
        50000, orig_dl, model, dims, device
    )
    mean_path, cov_path = calculate_activation_statistics(
        50000, synthetic_dl, model, dims, device
    )
    fid = calculate_frechet_distance(mean_ref, cov_ref, mean_path, cov_path)
    return fid


def compute_PSKL(original_loader, sample_loader, length):
    pskl_fft_mean = []
    pskl_pakt_mean = []
    for _ in range(length):
        orig_data, _ = next(iter(original_loader))
        sample_data = next(iter(sample_loader))
        pskl_fft = fourier_power_divergence(orig_data, sample_data)
        pskl_fft_mean.append(pskl_fft)
        pskl_pakt = wavelet_packet_power_divergence(orig_data, sample_data)
        pskl_pakt_mean.append(pskl_pakt)
    assert len(pskl_fft_mean) != 0
    assert len(pskl_pakt_mean) != 0
    return sum(pskl_fft_mean)/len(pskl_fft_mean), sum(pskl_pakt_mean)/len(pskl_pakt_mean)




def main():
    dataset_name = "CIFAR10"
    config_name = cifar10 if dataset_name.upper() == 'CIFAR10' else None
    data_path = "."
    bs = 50
    input_folder_path = None

    sampled_folder_path = '/home/lveerama/results/metrics_sampled_images/DDIM/sample_imgs_cifar10_32_DDIM/'
    sampled_data_path = os.path.join(sampled_folder_path, 'sample_imgs')
    if not os.path.isdir(sampled_data_path):
        npz_files = glob.glob(f'{sampled_folder_path}*.npz')
        if len(npz_files) == 0:
            raise ValueError("Please feed the folder with images individually or npz format")
        print("Found data as in npz files, extracting to the sample_imgs folder")
        sample_save_path = os.path.join(sampled_folder_path, 'sample_imgs')
        os.makedirs(sample_save_path, exist_ok=True)
        counter = 1
        for file in npz_files:
            batched_images = np.load(file)['x']
            for idx in range(len(batched_images)):
                fn = os.path.join(sample_save_path, f'{counter:06d}.png')
                im = Image.fromarray((batched_images[idx, :, :, :]*255.).astype(np.uint8))
                im.save(fn)
                counter += 1
    else:
        print(f"Found individual images in the folder {sampled_data_path}")
    
    normalize = transforms.Normalize(
        mean = config_name.dataset['mean'],
        std = config_name.dataset['std']
    )

    if dataset_name.upper() == "CIFAR10":
        input_transforms = False
        train_set = datasets.CIFAR10(
            "../cifar_data",
            download=True,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # normalize
                 ]
            ),)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        #normalize
    ])

    sampled_set = ImagePathDataset(data_path=sampled_data_path, transforms_=data_transforms)

    sampled_loader = torch.utils.data.DataLoader(
        sampled_set, batch_size = bs, shuffle=False, drop_last = False, num_workers=8
    )

    original_loader = torch.utils.data.DataLoader(
        train_set, batch_size = bs, shuffle=False, drop_last = False, num_workers=8
    )

        
    metrics = {}
    comp_feats = [64, 192] if dataset_name.upper() == 'CIFAR10' else [768, 2048]
    fid = {}
    for feat in comp_feats:
        fid_feat = compute_FID(original_loader, sampled_loader, "cuda", feat)
        print(fid_feat)
        fid[feat] = fid_feat
    metrics['FID'] = fid

    fft, packet = compute_PSKL(original_loader, sampled_loader, len(original_loader))
    metrics['PSKL_FFT'] = fft.item()
    metrics['PSKL_PACKET'] = packet.item()


    metrics_fn = os.path.join(sampled_folder_path, 'metrics.txt')
    with open(metrics_fn, 'w') as file:
        file.write(json.dumps(metrics))

    shutil.rmtree(sampled_data_path)

if __name__ == "__main__":
    main()
