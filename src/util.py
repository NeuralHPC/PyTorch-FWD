from typing import Tuple

import argparse
import jax
import struct
import numpy as np
import jax.numpy as jnp


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
