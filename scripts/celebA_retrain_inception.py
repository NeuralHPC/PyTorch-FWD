import PIL.Image
import numpy as np

import torch
import torchvision.transforms
import pandas
import PIL
from typing import Any
from torch import nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision

label_map = {pos: key for pos, key in enumerate([
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young'
])}

class CelebADataset(Dataset):
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.annos = np.array(pandas.read_csv(f"{data_dir}/Anno/list_attr_celeba.txt"))
        self.annos_dict = {row[0].split()[0]: row[0].split()[1:] for row in self.annos[1:]}

    def __len__(self):
        return len(self.annos_dict)

    def __getitem__(self, index) -> torch.Tensor:
        img_no_str = f"{index:6d}"
        img_path = f"{self.data_dir}/Img/img_align_celeba_png/{img_no_str}.jpg"
        with PIL.Image.open(img_path) as im:
            tensor = torchvision.transforms.ToTensor()(im)
        return {'img': tensor, 'anno': self.annos_dict[f"{img_no_str}.jpg"]}

if __name__ == "__main__":
    dataset = CelebADataset(data_dir='/home/wolter/Downloads/CelebA')

    net = torchvision.models.Inception3()
    net.fc = nn.Linear(2048, 39)
    # pass

    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for batch in loader:
        pass