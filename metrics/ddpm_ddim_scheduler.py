from diffusers import (
    UNet2DModel,
    DDPMParallelScheduler,
    DDIMParallelScheduler,
)
from typing import Dict, List, Any
import torch
import numpy as np
import sys
import argparse
import os
from tqdm import tqdm

model_id: Dict[str, List[Any]] = {
    "cifar10-32": ["google/ddpm-cifar10-32", 50000],
    "celebahq-256": ["google/ddpm-celebahq-256", 30000],
    "church-256": ["google/ddpm-church-256", 30000],
    "bedroom-256": ["google/ddpm-bedroom-256", 30000],
}


def main(scheduler_nm: str, dataset: str, input_shape: int) -> None:
    global model_id
    diffusion_steps = 1000
    batch_size = 1
    device = "cuda"
    diffusion_sampler = (
        DDIMParallelScheduler
        if scheduler_nm.upper() == "DDIM"
        else DDPMParallelScheduler
    )

    try:
        model_name, num_samples = model_id[f"{dataset.lower()}-{input_shape}"]
    except Exception as e:
        print("Model doesn't exist for given dataset and input shape.")
        print(f"Supported datasets with image shapes are {model_id.keys()}")
        sys.exit(0)

    scheduler = diffusion_sampler.from_pretrained(model_name)
    model = UNet2DModel.from_pretrained(model_name)
    model = torch.compile(model).to(device)

    scheduler.set_timesteps(diffusion_steps)

    sample_path = f"./sample_imgs_{dataset}_{input_shape}_{scheduler_nm}"
    os.makedirs(sample_path, exist_ok=True)
    print(f"Saving the sampled images at {sample_path}", flush=True)
    img_size = model.config.sample_size

    total_batches = num_samples // batch_size
    print(
        f"Overall {total_batches} number of batches needs to be processed.", flush=True
    )
    with torch.no_grad():
        for batch in range(total_batches):
            noise = torch.randn((batch_size, 3, img_size, img_size)).to(device)
            input = noise
            for t in tqdm(scheduler.timesteps):
                noisy_residual = model(input, t).sample
                prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
                input = prev_noisy_sample
            images = (input / 2 + 0.5).clamp(0, 1)
            np_images = images.cpu().permute(0, 2, 3, 1).numpy()
            fpath = os.path.join(sample_path, f"batch_{batch+1}.npz")
            np.savez(fpath, x=np_images)
            print(f"Saving the batch {fpath.split('/')[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="DDPM_DDIM_Scheduler",
        description="Sanple DDPM and DDIM on various datasets",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Select the dataset."
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        required=True,
        help="Input size either width or height.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="DDIM",
        help="Select between DDIM and DDPM sampling.",
    )
    args = parser.parse_args()
    scheduler = args.scheduler
    dataset = args.dataset
    input_shape = args.input_shape
    main(scheduler, dataset, input_shape)
