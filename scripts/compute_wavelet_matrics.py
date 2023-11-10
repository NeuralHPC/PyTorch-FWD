import numpy as np
import matplotlib.pyplot as plt
import torch
from src.freq_math import wavelet_packet_power_divergence
import pickle
import os
from tqdm import tqdm
from glob import glob
from PIL import Image
from torchvision import transforms

original_path = '/home/lveerama/results/metrics_sampled_images/curated_original/*.jpg'
sample_path = '/home/lveerama/results/metrics_sampled_images/best_curated/best_curated/curated_StyleGAN2/*.png'

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path_):
        self.files = glob(path_)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        img = Image.open(path).convert('RGB')
        return self.transforms(img), path

original_loader = torch.utils.data.DataLoader(ImageDataset(original_path), batch_size=1, num_workers=11, prefetch_factor=11)
sample_loader = torch.utils.data.DataLoader(ImageDataset(sample_path), batch_size=1, num_workers=11, prefetch_factor=11)

sample_files = []
original_files = []
wv_score = []
flag = True
for sample_img, sample_path in tqdm(sample_loader):
    row_wv = []
    sample_img = sample_img#.to('cuda:0')
    sample_files.append(sample_path)
    for original_img, original_path in original_loader:
        if flag:
            original_files.append(original_path)
        original_img = original_img#.to('cuda:0')
        a, b = wavelet_packet_power_divergence(sample_img, original_img, level=4)
        fwd = 0.5*(a.item() + b.item())
        row_wv.append(fwd)
    wv_score.append(row_wv)

wv_score = np.array(wv_score)
print(wv_score.shape)

wavelets = {
    'score': wv_score,
    'sample_files': sample_files,
    'original_files': original_files
}

with open('wavelet_scores_StyleGAN2.pickle', 'wb') as handle:
    pickle.dump(wavelets, handle, protocol=pickle.HIGHEST_PROTOCOL)

dict_ = {}
with open('wavelet_scores_StyleGAN2.pickle', 'rb') as handle:
    dict_ = pickle.load(handle)

wv_score = dict_['score'].T
sample_files = dict_['sample_files']
ref_files = dict_['original_files']
ref_files = [file_[0] for file_ in ref_files]
sample_files = [file_[0] for file_ in sample_files]

min_idx = np.argmin(wv_score, axis=1)
print(min_idx)
os.makedirs('./WPSKL_matched_imgs_StyleGAN2', exist_ok=True)
for idx, arg in enumerate(tqdm(min_idx)):
    ref_img = Image.open(ref_files[idx]).convert('RGB')
    ref_tensor = transforms.ToTensor()(ref_img).unsqueeze(0)
    sample_img = Image.open(sample_files[arg]).convert('RGB')
    sample_tensor = transforms.ToTensor()(sample_img).unsqueeze(0)
    ab, ba = wavelet_packet_power_divergence(ref_tensor, sample_tensor, level=4, wavelet='sym5')
    fwd = 0.5*(ab + ba)
    plt.figure()
    plt.subplot(121)
    plt.imshow(ref_img)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(sample_img)
    plt.axis('off')
    plt.suptitle(f'{round(wv_score[idx, arg], 2)},{round(fwd.item(), 2)}, \n {ref_files[idx]}, \n {sample_files[arg]}', fontsize=6)
    plt.savefig(f'./WPSKL_matched_imgs_StyleGAN2/img_{idx}.png', bbox_inches='tight', dpi=600)
    plt.close()