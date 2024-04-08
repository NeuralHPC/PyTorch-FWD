"""Compute FID and wavelet matrices for real-curated and sampled images."""

import os
import pickle
import time
from glob import glob
from random import shuffle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from scripts.fid.fid import calculate_frechet_distance
from scripts.fid.inception import InceptionV3
from src.freq_math import (
    forward_wavelet_packet_transform,
    wavelet_packet_power_divergence,
)

original_image_path = (
    "/home/lveerama/results/metrics_sampled_images/curated_original/*.jpg"
)
sampled_image_path = "/home/lveerama/results/metrics_sampled_images/celebahq_samples/DDIM/sample_imgs_torch/*.jpg"
sample_pickle_acts = "DDIM_acts.pickle"

bidx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([bidx])
model = model  # .to('cuda:0')
model.eval()


############################################################
##        Uncommment to compute activations of images     ##
############################################################
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.files = glob(path)
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        img = Image.open(path).convert("RGB")
        return self.transforms(img), path


def compute_activation(img_tensor, model):
    op1 = model(img_tensor)[0]
    onp1 = op1.squeeze(3).squeeze(2).cpu().numpy()
    return onp1


dataloader = torch.utils.data.DataLoader(
    ImageDataset(sampled_image_path), batch_size=1, num_workers=11, prefetch_factor=11
)

act_dict = {}
for imgs, files in tqdm(dataloader):
    imgs = imgs  # .to('cuda:0')
    with torch.no_grad():
        act = compute_activation(imgs, model)
        for idx, file in enumerate(files):
            act_dict[file] = act
with open(sample_pickle_acts, "wb") as fp:
    pickle.dump(act_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

#################################################################
#################################################################
#################################################################
#########        Compute FID Matrix              ################


def compute_fid(activation1, activation2):
    m1, s1 = np.mean(activation1, axis=0), np.cov(activation1, rowvar=False)
    m2, s2 = np.mean(activation2, axis=0), np.cov(activation2, rowvar=False)
    return calculate_frechet_distance(m1, s1, m2, s2)


ref_acts = None
sample_acts = None

with open("original_acts.pickle", "rb") as handle:
    ref_acts = pickle.load(handle)

with open(sample_pickle_acts, "rb") as handle:
    sample_acts = pickle.load(handle)

fid_matrix = []
for _, act1 in tqdm(sample_acts.items()):
    row_fid = []
    for _, act2 in ref_acts.items():
        row_fid.append(compute_fid(act1, act2))
    fid_matrix.append(row_fid)
fid_matrix = np.asarray(fid_matrix)
with open("DDIM_FID_matrix.npy", "wb") as fp:
    np.save(fp, fid_matrix)

ref_files, sample_files = [], []
for file, _ in ref_acts.items():
    ref_files.append(file)

for file, _ in sample_acts.items():
    sample_files.append(file)

df = pd.DataFrame(fid_matrix, columns=ref_files, index=sample_files)
df.to_csv("./DDIM_FID_matrix.csv")
#################################################################
#################################################################
#################################################################
fid_matrix = np.load("DDIM_FID_matrix.npy").T
print(fid_matrix.shape)

ref_files = []
sample_files = []
for file, _ in ref_acts.items():
    ref_files.append(file)

for file, _ in sample_acts.items():
    sample_files.append(file)

print(np.min(fid_matrix), np.max(fid_matrix), np.mean(fid_matrix))

# min_indices = np.argsort(fid_matrix, axis=1)
# topk = 5
# print(min_indices.shape)
# min_indices = min_indices[:, :topk]
# print(min_indices.shape)#
os.makedirs("./fid_matched_imgs_DDIM/", exist_ok=True)
# for x in tqdm(range(min_indices.shape[0])):
#     plt.figure()
#     plt.subplot(1, topk+1, 1)
#     plt.imshow(ref_img)
#     plt.axis('off')
#     for y in range(topk):
#         sample_img = Image.open(sample_files[min_indices[x, y]]).convert('RGB')
#         sample_tensor = transforms.ToTensor()(sample_img).unsqueeze(0)
#         ab, ba = wavelet_packet_power_divergence(ref_tensor, sample_tensor, level=4, wavelet='sym5')
#         fwd = 0.5*(ab + ba)
#         plt.subplot(1, topk+1, y+2)
#         plt.imshow(sample_img)
#         plt.axis('off')
#         plt.title(f'{round(fid_matrix[x, y], 2)},{round(fwd.item(), 2)}', fontsize=6)
#     plt.savefig(f'./fid_matched_imgs/img_{x}.png', bbox_inches='tight', dpi=600)
#     plt.close()

min_idx = np.argmin(fid_matrix, axis=1)
for idx, arg in enumerate(min_idx):
    ref_img = Image.open(ref_files[idx]).convert("RGB")
    ref_tensor = transforms.ToTensor()(ref_img).unsqueeze(0)
    sample_img = Image.open(sample_files[arg]).convert("RGB")
    sample_tensor = transforms.ToTensor()(sample_img).unsqueeze(0)
    ab, ba = wavelet_packet_power_divergence(
        ref_tensor, sample_tensor, level=4, wavelet="sym5"
    )
    fwd = 0.5 * (ab + ba)
    plt.figure()
    plt.subplot(121)
    plt.imshow(ref_img)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(sample_img)
    plt.axis("off")
    plt.suptitle(
        f"{round(fid_matrix[idx, arg], 2)},{round(fwd.item(), 2)}, \n {ref_files[idx]}, \n {sample_files[arg]}",
        fontsize=6,
    )
    plt.savefig(f"./fid_matched_imgs_DDIM/img_{idx}.png", bbox_inches="tight", dpi=600)
    plt.close()
