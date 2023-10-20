import json
import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from PIL import Image


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
        # img = io.read_image(img_path)
        img = Image.open(img_path)
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
    dataset_name: str,
    batch_size: int,
    val_size: int,
    data_path: str = None,
    only_datasets: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Get the dataloaders based on the dataset.

    Args:
        dataset_name (str): Name of the dataset
        batch_size (int): Batch size
        data_path (str): Path to dataset. Defaults to None.
        only_datasets (bool): Return only datasets. Defaults to True.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing train and validation dataloaders/datasets
    """
    train_set, val_set = None, None
    if "mnist" in dataset_name.lower():
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
    elif "celebahq" in dataset_name.lower():
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
    elif "cifar10" in dataset_name.lower():
        # TODO: Get these normalize values from the config file later
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
        )

        train_set = datasets.CIFAR10(
            "../cifar_data",
            download=True,
            train=True,
            transform=transforms.Compose(
                [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
            ),
        )
        val_set = datasets.CIFAR10(
            "../cifar_data",
            download=True,
            train=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
    elif "celeba" in dataset_name.lower():
        raise NotImplementedError

    if only_datasets:
        return train_set, val_set

    trainloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=48
    )
    valloader = DataLoader(val_set, batch_size=val_size, shuffle=False, num_workers=48)
    return trainloader, valloader


def get_distributed_dataloader(
    dataset: Dataset,
    world_size: int,
    global_seed: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, torch.utils.data.Sampler]:
    """Return the distributed dataloader.

    Args:
        dataset (Dataset): Dataset object
        world_size (int): Total number of available GPUs
        global_seed (int): Overall seed
        batch_size (int): Batch size
        num_workers (int): Total number of workers in each node

    Returns:
        Tuple[DataLoader, torch.utils.data.Sampler]: Tuple containing dataloader and datasampler
    """
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, shuffle=True, seed=global_seed
    )
    return (
        DataLoader(
            dataset,
            batch_size=batch_size // world_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        sampler,
    )
