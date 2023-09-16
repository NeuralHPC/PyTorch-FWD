import json
import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, io, transforms


class CelebAHQDataset(Dataset):
    """CelebAHQ dataset."""

    def __init__(self, data_path: str, transform: transforms = None) -> None:
        """Initialize the CelebAHQ dataset.

        Args:
            data_path (str): Data path
            transform (transforms, optional): Data transforms. Defaults to None.
        """
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.labels = self.get_labels("./data/CelebAHQ/identity_CelebAHQ.txt")
        self.img_names = os.listdir(data_path)

    def get_labels(self, path: str) -> Dict[str, int]:
        """Ĺoad image labels.

        Args:
            path (str): Labels path

        Returns:
            Dict[str, int]: Dictionary containing image name and label
        """
        hq_labels = json.load(open(path, "r"))
        for key, value in hq_labels.items():
            value = int(value.replace("\n", ""))
            hq_labels[key] = value
        return hq_labels

    def __len__(self) -> int:
        """Total number of images.

        Returns:
            int: _Length of images
        """
        return len(self.img_names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Load each image and its correspondng label.

        Args:
            index (int): Index of the image

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing image and its label
        """
        img_path = os.path.join(self.data_path, self.img_names[index])
        img = io.read_image(img_path)
        if self.transform:
            img = self.transform(img)
        label = self.labels[self.img_names[index]]
        return img, label


class CelabADataset(Dataset):
    """CelebA dataset."""

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError


def get_dataloaders(
    dataset_name: str, batch_size: int, data_path: str = None
) -> Tuple[DataLoader, DataLoader]:
    """Get the dataloaders based on the dataset.

    Args:
        dataset_name (str): Name of the dataset
        batch_size (int): Batch size
        data_path (str, optional): Path to dataset. Defaults to None.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing train and validation dataloaders
    """
    train_set, val_set = None, None
    if dataset_name.lower() == "mnist":
        train_set = datasets.MNIST(
            "../mnist_data",
            download=True,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        )
        val_set = datasets.MNIST(
            "../mnist_data",
            download=True,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        )
    elif dataset_name.lower() == "celebahq":
        train_set = CelebAHQDataset(
            data_path,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        )
        val_set = CelebAHQDataset(
            data_path,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        )
    elif dataset_name.lower() == "celeba":
        raise NotImplementedError

    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_set, batch_size=1, shuffle=False)
    return trainloader, valloader
