from typing import Tuple, List, Optional
from multiprocessing.pool import ThreadPool


from glob import glob
import argparse
import jax
import struct
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from PIL import Image

def _parse_args():
    """Parse cmd line args for training an image classifier."""
    parser = argparse.ArgumentParser(description="Run a diffusion model")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="input batch size for testing (default: 30)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="learning rate for optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--epochs", type=int, default=21, help="number of epochs (default: 21)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="initial seed value (default:42)"
    )
    parser.add_argument(
        "--time-steps", type=int, default=40, help="steps per diffusion"
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
    
    
    return parser.parse_args()

def get_mnist_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """Return the mnist test data set in numpy arrays.

    Returns:
        (array, array): A touple containing the test
        images and labels.
    """
    with open("./data/MNIST/raw/t10k-images-idx3-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.array(np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">")))
        img_data_test = data.reshape((size, nrows, ncols))

    with open("./data/MNIST/raw/t10k-labels-idx1-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        lbl_data_test = np.array(np.fromfile(f, dtype=np.dtype(np.uint8)))
    return img_data_test, lbl_data_test


def get_mnist_train_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the mnist training data set.

    Returns:
        (array, array): A touple containing the training
        images and labels.
    """
    with open("./data/MNIST/raw/train-images-idx3-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.array(np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">")))
        img_data_train = data.reshape((size, nrows, ncols))

    with open("./data/MNIST/raw/train-labels-idx1-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        lbl_data_train = np.array(np.fromfile(f, dtype=np.dtype(np.uint8)))
    return img_data_train, lbl_data_train


def get_batched_celebA_paths(batch_size: int) -> List[np.ndarray]:
    img_folder_path = '/home/wolter/uni/diffusion/data/celebA/CelebA/Img/img_align_celeba_png/img_align_celeba_png'
    # TODO: implement a train/test split.
    partition_list = '/home/wolter/uni/diffusion/data/celebA/CelebA/list_eval_partition.txt'

    image_path_list_array = np.array(glob(img_folder_path + '/*.png'))
    image_count = len(image_path_list_array)
    image_path_batches = np.array_split(image_path_list_array, image_count//batch_size )

    return image_path_batches


def batch_loader(batch_array: np.ndarray) -> np.ndarray:
    # load a single image batch into memory.

    def load(path: str) -> np.ndarray:
        return np.array(Image.open(path))
    arrays = list(map(load, batch_array))
    # arrays = pool.map(load, batch_array, 64//4)
    return np.stack(arrays)


def multi_batch_loader(batch_list):
    # batches = list(map(batch_loader, batch_list))
    with ThreadPool() as p:
        batches = p.map(batch_loader, batch_list, chunksize=5)
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