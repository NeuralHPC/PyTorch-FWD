import argparse
import os
from typing import List, Optional

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np


def _get_local_rank():
    """Return local rank."""
    return int(os.environ["LOCAL_RANK"])


def _get_global_rank():
    """Retrun global rank."""
    return int(os.environ["RANK"])


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
        "--lr",
        type=float,
        default=None,
        help="User defined learning rate (default: lr specified in config file)",
    )
    parser.add_argument(
        "--epochs", type=int, default=51, help="number of epochs (default: 21)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="initial seed value (default:42)"
    )
    parser.add_argument(
        "--time-steps", type=int, default=40, help="steps per diffusion"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save the model for every specified epochs",
    )
    parser.add_argument(
        "--print-every", type=int, default=100, help="Print every specified step"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Saved model path in case of resuming the training",
    )
    parser.add_argument(
        "--allow-tf32", help="Use tensorflot32 operations.", action="store_true"
    )
    parser.add_argument(
        "--loss-type",
        default="MSE",
        choices=["MSE", "PACKET", "MIXED"],
        help="Choice of loss function",
    )
    parser.add_argument(
        "--packet-norm-type",
        type=str,
        default=None,
        choices=[None, "log", "weighted"],
        help="Provide packet norm type only when using packet or mixed loss.",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="sym5",
        choices=["Haar", "sym5"],
        help="Type of wavelet for packet loss computation.",
    )
    parser.add_argument(
        "--max-level", type=int, default=2, help="Depth of wavelet decomposition."
    )
    parser.add_argument(
        "--loss-sigma", type=float, default=0.3, help="Weigting factor for mixed loss."
    )
    parser.add_argument("--logdir", type=str, default="./log", help="logdir name.")
    parser.add_argument(
        "--clip-grad-norm", type=float, default=0.0, help="Gradient clipping value."
    )
    parser.add_argument(
        "--distribute", help="Use for multinode training.", action="store_true"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=["CIFAR10", "CELEBA64", "CELEBAHQ64", "CELEBAHQ128"],
        help="Select the dataset to diffuse",
    )
    return parser.parse_args()


def _sampler_args():
    parser = argparse.ArgumentParser(
        prog="Image sampler", description="Sample Images from the diffusion model"
    )
    parser.add_argument("--ckpt-path", required=True, help="Checkpoint path")
    parser.add_argument(
        "--input-shape", type=int, required=True, help="Sampled input shape"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed value")
    parser.add_argument(
        "--diff-steps", type=int, default=1000, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=["CIFAR10", "CELEBA64", "CELEBAHQ64", "CELEBAHQ128"],
        help="Select the dataset to diffuse",
    )
    parser.add_argument(
        "--allow-tf32", help="Use tensorflot32 operations.", action="store_true"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="input batch size for testing (default: 50)",
    )
    parser.add_argument(
        "--distribute", help="Use for multinode training.", action="store_true"
    )
    parser.add_argument(
        "--sampler",
        default="DDPM",
        choices=["DDPM", "DDIM"],
        help="Use DDIM Sampling else DDPM is used by default",
    )
    return parser.parse_args()


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

    metadata = dict(title="Diffusion", artist="Matplotlib", comment="Diffusion movie!")
    writer = ffmpeg_writer(fps=60, metadata=metadata)

    fig = plt.figure()
    l = plt.imshow(images[0] / np.max(np.abs(images[0])))
    plt.colorbar()

    # plt.xlim(-xlim, xlim)
    # plt.ylim(-ylim, ylim)

    with writer.saving(fig, f"{name}.gif", 100):
        for img in images:
            l.set_data(img / np.max(np.abs(img)))
            writer.grab_frame()


# def _save_model(
#     checkpoint_dir: str,
#     time: datetime.datetime,
#     epochs: int,
#     model_data: Tuple[FrozenDict],
# ) -> None:
#     """Save the model parameters

#     Args:
#         checkpoint_dir (str): Checkpoint directory
#         time (datetime.datetime): Time of saving
#         epochs (int): Epoch of saving
#         model_data (Tuple[FrozenDict]): Tuple containing the model data
#     """
#     save_dir = os.path.join(checkpoint_dir, f"e_{epochs}_time_{time}.pkl")
#     with open(save_dir, "wb") as fp:
#         pickle.dump(model_data, fp)
