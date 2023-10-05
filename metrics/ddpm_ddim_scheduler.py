import torch
import os
import sys
import math
import argparse
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from diffusers import DDIMScheduler, UNet2DModel, DDPMScheduler
import numpy as np


model_id = {
    "cifar10-32": ["google/ddpm-cifar10-32", 50000],
    "celebahq-256": ["google/ddpm-celebahq-256", 30000],
    "church-256": ["google/ddpm-church-256", 30000],
    "bedroom-256": ["google/ddpm-bedroom-256", 30000],
}


class Trainer:
    def __init__(
        self, model, global_seed, scheduler, img_size, batch_size, sample_path
    ):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank])
        self.seed = global_seed + self.global_rank
        self.scheduler = scheduler
        self.noise_img_shape = (batch_size, 3, img_size, img_size)
        self.sample_path = sample_path

    def sample(self, total_batches):
        with torch.no_grad():
            for batch in range(total_batches):
                noise = torch.randn(self.noise_img_shape).to(self.local_rank)
                input = noise
                for t in tqdm(self.scheduler.timesteps):
                    noisy_res = self.model(input, t).sample
                    prev_noisy_sample = self.scheduler.step(
                        noisy_res, t, input
                    ).prev_sample
                    input = prev_noisy_sample
                images = (input / 2 + 0.5).clamp(0, 1)
                np_images = images.cpu().permute(0, 2, 3, 1).numpy()
                # print(f"[GPU: {self.global_rank}], Img shape: {np_images.shape}")
                str_time = str(time.time()).replace(".", "_")
                fname = f"batch_{str_time}.npz"
                fpath = os.path.join(self.sample_path, fname)
                np.savez(fpath, x=np_images)
                print(f"Saved batch at {fname}", flush=True)


def main(scheduler_nm: str, dataset: str, input_shape: int):
    global model_id
    torch.backends.cuda.matmul.allow_tf32 = True
    init_process_group(backend="nccl")

    try:
        model_name, num_samples = model_id[f"{dataset.lower()}-{input_shape}"]
    except Exception as e:
        print("Model doesn't exist for given dataset and input shape.")
        print(f"Supported datasets with image shapes are {model_id.keys()}")
        sys.exit(0)
    batch_size = 50

    diffusion_sampler = (
        DDIMScheduler if scheduler_nm.upper() == "DDIM" else DDPMScheduler
    )
    scheduler = diffusion_sampler.from_pretrained(model_name)
    model = UNet2DModel.from_pretrained(model_name)
    scheduler.set_timesteps(1000)
    total_batches = int(
        math.ceil(num_samples / (batch_size * torch.distributed.get_world_size()))
    )
    global_rank = int(os.environ["RANK"])
    sample_path = f"./sample_imgs_{dataset}_{input_shape}_{scheduler_nm}"
    os.makedirs(sample_path, exist_ok=True)
    if global_rank == 0:
        print(f"Running {total_batches} number of batches for sampling", flush=True)
        print(f"Saving the sampled images at {sample_path}", flush=True)

    trainer = Trainer(
        model, 0, scheduler, model.config.sample_size, batch_size, sample_path
    )
    trainer.sample(total_batches)
    destroy_process_group()


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
