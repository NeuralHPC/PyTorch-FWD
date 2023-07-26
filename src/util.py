from typing import Tuple, List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor

from glob import glob
import argparse
import jax
import struct
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import pandas as pd
import os



from PIL import Image

def _parse_args():
    """Parse cmd line args for training an image classifier."""
    parser = argparse.ArgumentParser(description="Run a diffusion model")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="input batch size for testing (default: 50)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="learning rate for optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--epochs", type=int, default=51, help="number of epochs (default: 21)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="initial seed value (default:42)"
    )
    parser.add_argument(
        "--time-steps", type=int, default=50, help="steps per diffusion"
    )
    parser.add_argument(
        "--gpus", type=int, default=-1, help="set gpu no by hand. Use all if -1 (default)."
    )
    parser.add_argument(
        "--logdir", type=str, default="./log", help="logdir name."
    )
    parser.add_argument(
        "--distribute", help="TODO: Use for multinode training.", action='store_true'
    )
    parser.add_argument(
        "--data-dir", required=True, help="Base dataset path"
    )
    parser.add_argument(
        "--wavelet-loss", help="Use wavelets fix high frequency artifacts.", action='store_true'
    )
    return parser.parse_args()


def get_mnist_test_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return the mnist test data set in numpy arrays.

    Returns:
        (array, array): A tuple containing the test
        images and labels.
    """
    test_imgs_path = os.path.join(data_dir, 'raw/t10k-images-idx3-ubyte')
    test_lbls_path = os.path.join(data_dir, 'raw/t10k-labels-idx1-ubyte')
    
    with open(test_imgs_path, "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.array(np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">")))
        img_data_test = data.reshape((size, nrows, ncols))

    with open(test_lbls_path, "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        lbl_data_test = np.array(np.fromfile(f, dtype=np.dtype(np.uint8)))
    return img_data_test, lbl_data_test


def get_mnist_train_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the mnist training data set.

    Returns:
        (array, array): A tuple containing the training
        images and labels.
    """
    train_imgs_path = os.path.join(data_dir, 'raw/train-images-idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'raw/train-labels-idx1-ubyte')

    with open(train_imgs_path, "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.array(np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">")))
        img_data_train = data.reshape((size, nrows, ncols))

    with open(train_labels_path, "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        lbl_data_train = np.array(np.fromfile(f, dtype=np.dtype(np.uint8)))
    return img_data_train, lbl_data_train



def get_batched_celebA_paths(data_dir: str, batch_size: int = 50, split: str = 'train') -> List[np.ndarray]:
    # img_folder_path = '/home/wolter/uni/diffusion/data/celebA/CelebA/Img/img_align_celeba_png/img_align_celeba_png'
    img_folder_path = os.path.join(data_dir, 'Img/img_align_celeba/')
    # partition_list = '/home/wolter/uni/diffusion/data/celebA/CelebA/Eval/list_eval_partition.txt'
    partition_path = os.path.join(data_dir, 'Eval/list_eval_partition.txt')
    labels_path = os.path.join(data_dir, 'Anno/identity_CelebA.txt')
    partition_df = pd.read_csv(partition_path, names=['images', 'split'], sep=' ')

    split_val = 0
    if split == 'validation':
        split_val = 1

    image_names = []
    image_names = partition_df[partition_df['split'] == split_val]['images']
    image_names = [image_name.split('.')[0] + ".jpg" for image_name in image_names]
    image_path_list_array = np.array([os.path.join(img_folder_path, image_name) for image_name in image_names])
    
    image_count = len(image_path_list_array)
    image_path_batches = np.array_split(image_path_list_array, image_count//batch_size )

    labels_dict = get_label_dict(labels_path)
    return image_path_batches, labels_dict



def get_label_dict(path: str) -> Dict[str, int]:
    # '/home/wolter/uni/diffusion/data/celebA/CelebA/Anno/identity_CelebA.txt'
    labels_dict = {}
    with open(path, 'r') as fp:
        for line in fp:
            line = line.replace('\n', '')
            key, value = line.split(' ')
            key = key.split('.')[0]
            labels_dict[key] = int(value)
    return labels_dict


def batch_loader(batch_array: np.ndarray, labels_dict: Dict[str, int],
                 resize: Tuple[int, int] = (64, 64)) -> np.ndarray:
    # load a single image batch into memory.

    def load(path: str) -> np.ndarray:
        img = Image.open(path)
        img = img.resize(resize, Image.Resampling.LANCZOS)
        return img

    def label(path: str) -> np.ndarray:
        img_name = path.split("/")[-1].split(".")[0]
        return labels_dict[img_name]

    with ThreadPoolExecutor(max_workers=8) as p:
        arrays = list(p.map(load, batch_array))
        labels = list(p.map(label, batch_array))
    return np.stack(arrays), np.stack(labels)


def multi_batch_loader(batch_list):
    # batches = list(map(batch_loader, batch_list))
    with ThreadPoolExecutor() as p:
        batches = p.map(batch_loader, batch_list, chunksize=10)
    return batches



def write_movie(
    images: List[np.ndarray],
    name: Optional[str] = "diff_movie",
    xlim: Optional[int] = 3,
    ylim: Optional[int] = 3,
):
    """Write the optimization steps into a mp4-movie file.

    Args:
        images (list): A list with diffusion steps.
        name (str, optional): The name of the movie file. Defaults to "grad_movie".
        xlim (int, optional): Largest x value in the data. Defaults to 3.
        ylim (int, optional): Largest y value in the data. Defaults to 3.
    
    Raises:
        RuntimeError: If conda ffmpeg package is not installed.
    """
    try:
        ffmpeg_writer = manimation.writers["ffmpeg"]
    except RuntimeError:
        raise RuntimeError(
            "RuntimeError: If you are using anaconda or miniconda there might "
            "be a missing package named ffmpeg. Try installing it with "
            "'conda install -c conda-forge ffmpeg' in your terminal."
        )

    metadata = dict(
        title="Diffusion", artist="Matplotlib", comment="Diffusion movie!"
    )
    writer = ffmpeg_writer(fps=3, metadata=metadata)

    fig = plt.figure()
    l = plt.imshow(images[0]/np.max(np.abs(images[0])))
    plt.colorbar()

    # plt.xlim(-xlim, xlim)
    # plt.ylim(-ylim, ylim)

    with writer.saving(fig, f"{name}.gif", 100):
        for img in images:
            l.set_data(img/np.max(np.abs(img)))
            writer.grab_frame()

