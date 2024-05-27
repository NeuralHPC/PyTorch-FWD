import numpy as np
import glob
import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scripts.celebA_retrain_inception import label_map
from scripts.fid.fid import calculate_frechet_distance

import pytorch_fid.fid_score
import PIL

from tqdm import tqdm

class SimpleLoader(Dataset):

    def __init__(self, data_dir, type='jpg'):
        self.data_dir = data_dir
        self.file_list = glob.glob(data_dir + f"/*.{type}")
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((299, 299)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([x / 255.0 for x in [129.058, 108.485, 97.622]],
                                             [x / 255.0 for x in [78.338, 73.131, 72.970]])
            ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index) -> torch.Tensor:
        # img 0 does not exist.
        img_path = self.file_list[index]
        with PIL.Image.open(img_path) as im:
            tensor = self.transforms(im)
        return {'img': tensor}

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


if __name__ == '__main__':
    net = torchvision.models.Inception3()
    net.fc = nn.Linear(2048, 40)
    net = nn.DataParallel(net.cuda())

    print('loading pretrained network')
    net_weights = torch.load('../weights/inception_converged_celeba/inception_converged_99.pth')

    # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    # model = InceptionV3([block_idx])

    clean_weights = {}
    for key, weight in net_weights.items():
        new_key = ".".join(key.split('.')[1:])
        clean_weights[new_key] = weight

    missing, unexpected = net.load_state_dict(clean_weights)
    print("missing", missing, "unxepected", unexpected)

    net.module.fc = Identity()
    # net.module.avgpool = Identity()
    net.eval()

    # celebA - acc
    # data = ImageLoader('../data/celeba/CelebA')



    def get_mu_sigma(data_loader):
        acts = []
        for img_batch in tqdm(data_loader):
            with torch.no_grad():
                out = net(img_batch['img'].cuda())
            acts.append(out.cpu().numpy())

        acts_stack = np.concatenate(acts)
        mu = np.mean(acts_stack, axis=0)
        sigma = np.cov(acts_stack, rowvar=False)
        return mu, sigma

    # agri-data
    original_loader =  DataLoader(SimpleLoader('../data/agri/dataset256/images/'), batch_size=128, num_workers=6)
    ddgan_loader = DataLoader(SimpleLoader('../data/agri/generated_samples/agri'), batch_size=128, num_workers=6)
    pggan_loader = DataLoader(SimpleLoader('../data/agri/out', type='png'), batch_size=128, num_workers=6)

    agri_mu, agri_sigma = get_mu_sigma(original_loader)
    ddgan_mu, ddgan_sigma = get_mu_sigma(ddgan_loader)
    pggan_mu, pggan_sigma = get_mu_sigma(pggan_loader)

    fid_agri_ddgan = pytorch_fid.fid_score.calculate_frechet_distance(agri_mu, agri_sigma, ddgan_mu, ddgan_sigma)
    fid_agri_pggan = pytorch_fid.fid_score.calculate_frechet_distance(agri_mu, agri_sigma, pggan_mu, pggan_sigma)

    print(f"Fid agri-DDGAN: {fid_agri_ddgan}, FID agri-PGGAN: {fid_agri_pggan}")

    # celebA-HQ
    original_loader =  DataLoader(SimpleLoader('../data/celeba_hq/celeba_hq_256'), batch_size=128, num_workers=6)
    ddgan_loader = DataLoader(SimpleLoader('../data/celeba_hq/DDGAN/celeba_256'), batch_size=128, num_workers=6)
    pggan_loader = DataLoader(SimpleLoader('../data/celeba_hq/pggan_celebahq_generated', type='png'), batch_size=128, num_workers=6)

    celebA_mu, celebA_sigma = get_mu_sigma(original_loader)
    ddgan_mu, ddgan_sigma = get_mu_sigma(ddgan_loader)
    pggan_mu, pggan_sigma = get_mu_sigma(pggan_loader)

    fid_celebA_ddgan = pytorch_fid.fid_score.calculate_frechet_distance(celebA_mu, celebA_sigma, ddgan_mu, ddgan_sigma)
    fid_celebA_pggan = pytorch_fid.fid_score.calculate_frechet_distance(celebA_mu, celebA_sigma, pggan_mu, pggan_sigma)

    print(f"Fid CelebA-DDGAN: {fid_celebA_ddgan}, FID CelebA-PGGAN: {fid_celebA_pggan}")

    breakpoint()

