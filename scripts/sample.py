"""Sample based on the dataset."""
import os

import torch
import torch.distributed as dist

from scripts.train import load_config, instantiate_model
from src.dataloader import get_dataloaders, get_distributed_dataloader
from src.sampler import Sampler
from src.util import _sampler_args


def main():
    global args
    config = load_config(args.dataset)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dist.init_process_group(backend='nccl')
    torch.manual_seed(args.seed)
    model = instantiate_model(config.model_config)

    data_set, _ = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        val_size=20,
        data_path=config.data_dir,
        only_datasets=True
    )
    dataloader, _ = get_distributed_dataloader(
        dataset=data_set,
        world_size=dist.get_world_size(),
        global_seed=args.seed,
        batch_size=args.batch_size,
        num_workers=config.dataset_config["num_workers"]
    )
    save_path = f"./sample_imgs_{args.dataset}_{args.sampler}/"
    os.makedirs(save_path, exist_ok=True)

    sampler = Sampler(
        model=model,
        args=args,
        save_path=save_path,
        std=config.dataset_config["std"],
        mean=config.dataset_config["mean"]
    )
    sampler.conditional_sample(batched_labels=dataloader)


if __name__ == "__main__":
    args = _sampler_args()
    main()
