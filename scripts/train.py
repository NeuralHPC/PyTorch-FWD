import datetime
import os
from typing import Dict, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import celeba, cifar10
from src.dataloader import get_dataloaders, get_distributed_dataloader
from src.improved_UNet import Improv_UNet
from src.nn import get_loss_fn
from src.trainer import Trainer
from src.util import _parse_args, _get_global_rank, _get_local_rank


def load_config(dataset_name: str) -> Any:
    """Load the configurations from config folder.

    Args:
        dataset_name (str): Name of the dataset

    Returns:
        Any: Class containing the configuration details.
    """
    dataset_name = dataset_name.upper()
    config_module = cifar10 if dataset_name == 'CIFAR10' else celeba
    config = getattr(config_module, dataset_name)()
    return config


def instantiate_model(model_config: Dict[str, Any]) -> nn.Module:
    """Instantiate the UNet model.

    Args:
        model_config (Dict[str, Any]): Dictionary containing the model configuration details.

    Returns:
        nn.Module: Instantiate the model.
    """
    attn_res = []
    for value in model_config["attn_res"]:
        attn_res.append(model_config["input_size"] // int(value))

    model = Improv_UNet(
        in_channels=model_config["in_c"],
        model_channels=model_config["model_c"],
        out_channels=model_config["out_c"],
        num_res_blocks=model_config["num_res_blocks"],
        attention_resolutions=attn_res,
        dropout=model_config["dropout"],
        channel_mult=model_config["channel_mult"],
        num_classes=model_config["num_classes"],
        num_heads=model_config["num_heads"],
        num_heads_upsample=model_config["num_heads_ups"],
        use_scale_shift_norm=model_config["use_scale_shift_norm"]
    )
    return model


def main():
    """DDP Training for DDPM/DDIM diffusion models."""
    global args
    config = load_config(args.dataset)
    if args.allow_tf32:
        print("TF32 matmul are activated. \
              (Note: This improves training on Ampere GPU devices but affects accuracy a little.)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize the models
    dist.init_process_group(backend='nccl')
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    local_rank = _get_local_rank()
    global_rank = _get_global_rank() if args.distribute else local_rank
    model = instantiate_model(config.model_config)

    if config.data_dir is None:
        raise ValueError('Datapath is None, please set the datapath in corresponding config file.')
    if not os.path.exists(config.data_dir):
        raise ValueError('Data directory doesnot exist, please provide proper path in corresponding config file.')

    # Dataloading
    train_set, val_set = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        val_size=20,
        data_path=config.data_dir,
        only_datasets=True
    )
    train_loader, train_sampler = get_distributed_dataloader(dataset=train_set,
                                                             world_size=dist.get_world_size(),
                                                             global_seed=args.seed,
                                                             batch_size=args.batch_size,
                                                             num_workers=config.dataset_config["num_workers"])
    val_loader, _ = get_distributed_dataloader(dataset=val_set,
                                               world_size=dist.get_world_size(),
                                               global_seed=args.seed,
                                               batch_size=args.batch_size,
                                               num_workers=config.dataset_config["num_workers"])

    # model utils intialization
    loss_fn = get_loss_fn(args.loss_type)
    args.clip_grad_norm = config.optimizer_config["clip_grad_norm"] if args.clip_grad_norm == 0 else args.clip_grad_norm
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer_config["lr"])
    writer = None
    save_path = None
    if global_rank == 0:
        dt_now = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-").replace(".", "-")
        writer_name = f"_dataset_{args.dataset}_loss_{args.loss_type}_on_{dt_now}"
        save_path = f'./logs/{args.dataset}_{args.loss_type}_{dt_now}/'
        if args.model_path is not None:
            save_path = "/".join(args.model_path.split('/')[:-1])
        writer = SummaryWriter(log_dir=save_path, comment=writer_name)

    # Trainer Loop
    trainer = Trainer(
        model=model,
        args=args,
        optimizer=optimizer,
        loss_fn=loss_fn,
        writer=writer,
        save_path=save_path
    )
    trainer.train(
        max_epochs=args.epochs,
        dataloader=train_loader,
        sampler=train_sampler
    )
    if global_rank == 0:
        writer.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    args = _parse_args()
    print(args)
    main()
