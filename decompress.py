"""Decompress npz image files."""

import torchvision
import torch
import glob, os
import numpy as np
from PIL import Image
from tqdm import tqdm

folder_path = './sample_CIFAR10_DDPM_MSE/'
sample_path = os.path.join(folder_path, 'sample_imgs_torch')
min_, max_ = [], []
if not os.path.isdir(sample_path):
    npz_files = glob.glob(f'{folder_path}*.npz')
    os.makedirs(sample_path, exist_ok=True)
    counter = 1
    for file in npz_files:
        batched_imgs = np.load(file)['x']
        batched_imgs = torch.from_numpy(batched_imgs).permute(0, 3, 1, 2)
        batched_imgs = torch.clamp(batched_imgs, 0, 1) 
        for ix in tqdm(range(len(batched_imgs))):
            fn = os.path.join(sample_path, f"{counter:06d}.jpg")
            torchvision.utils.save_image(
                batched_imgs[ix, :, :, :], fn
            )
            counter += 1

else:
    print('File already exists')
