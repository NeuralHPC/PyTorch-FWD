import argparse
import datetime
import os
import pickle
from typing import List, Optional, Tuple

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from flax.core.frozen_dict import FrozenDict


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
        "--max-workers",
        type=int,
        default=16,
        help="the number of data loading workers.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="learning rate for optimizer (default: 1e-4)",
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
        "--gpus",
        type=int,
        default=-1,
        help="set gpu no by hand. Use all if -1 (default).",
    )
    parser.add_argument("--logdir", type=str, default="./log", help="logdir name.")
    parser.add_argument(
        "--distribute", help="TODO: Use for multinode training.", action="store_true"
    )
    parser.add_argument("--data-dir", required=True, help="Base dataset path")
    parser.add_argument("--resize", type=int, default=64, help="Resize the input image")
    parser.add_argument(
        "--channel-mult",
        type=str,
        default="1,2,2,4",
        help="Channel multiplier for the UNet",
    )
    parser.add_argument(
        "--num-res-blocks",
        type=int,
        default=2,
        help="Number of residual blocks for the model",
    )
    parser.add_argument(
        "--conditional", action="store_false", help="Add the class condition to model"
    )
    parser.add_argument(
        "--attn-heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--attn-heads-upsample",
        type=int,
        default=-1,
        help="Number of attention heads during upsample",
    )
    parser.add_argument(
        "--attn-resolution",
        type=str,
        default="16,8",
        help="Resolutions at which attention should be applied",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=128,
        help="Base channels for the UNet to start with",
    )
    parser.add_argument(
        "--wavelet-loss",
        help="Use wavelets fix high frequency artifacts.",
        action="store_true",
    )
    parser.add_argument(
        "--dataset", type=str, default="CelebAHQ", help="Select the dataset to diffuse"
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
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
        "--diff-steps", type=int, default=40, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--gif", action="store_true", help="Store diffusion process as a GIF"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=-1,
        help="set gpu no by hand. Use all if -1 (default).",
    )
    parser.add_argument(
        "--use-DDIM",
        action="store_true",
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


def _save_model(
    checkpoint_dir: str,
    time: datetime.datetime,
    epochs: int,
    model_data: Tuple[FrozenDict],
) -> None:
    """Save the model parameters

    Args:
        checkpoint_dir (str): Checkpoint directory
        time (datetime.datetime): Time of saving
        epochs (int): Epoch of saving
        model_data (Tuple[FrozenDict]): Tuple containing the model data
    """
    save_dir = os.path.join(checkpoint_dir, f"e_{epochs}_time_{time}.pkl")
    with open(save_dir, "wb") as fp:
        pickle.dump(model_data, fp)
