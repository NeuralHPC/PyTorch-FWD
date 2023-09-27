"""Compute FID Score based on the sampled and original dataset."""

import torch
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np
import json
from torchvision import transforms

from src.dataloader import get_dataloaders


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = self.files[i]
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
        batch -= batch.min()
        batch *= 255 / batch.max()
        batch = batch.int().float() / 255
        if start_idx == 0:
            print(type(batch))
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
    print(block_idx)
    model = InceptionV3([block_idx]).to(device)
    mean_ref, cov_ref = calculate_activation_statistics(
        50000, orig_dl, model, dims, device
    )
    mean_path, cov_path = calculate_activation_statistics(
        50000, synthetic_dl, model, dims, device
    )
    fid = calculate_frechet_distance(mean_ref, cov_ref, mean_path, cov_path)
    return fid


def main():
    dataset_name = "CIFAR10"
    data_path = "."
    train_loader, _ = get_dataloaders(dataset_name, 500, 2000, data_path)
    orig_data_lst = []
    for input_, _ in train_loader:
        orig_data_lst.append(input_)
    orig_data = torch.concatenate(orig_data_lst, dim=0)
    print(orig_data.shape)

    sampled_data = torch.load(
        "./sample_imgs/sampled_tensors.pt", map_location=torch.device("cpu")
    )
    # rescale_fn = lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    # sampled_data = torch.stack([rescale_fn(img) for img in sampled_data], dim=0)
    print(sampled_data.shape)

    sampled_dl = torch.utils.data.DataLoader(
        ImagePathDataset(sampled_data, None), batch_size=500
    )
    orig_dl = torch.utils.data.DataLoader(
        ImagePathDataset(orig_data, None), batch_size=500
    )

    fid = compute_FID(orig_dl, sampled_dl, "cuda", 2048)
    print(fid)


if __name__ == "__main__":
    main()
