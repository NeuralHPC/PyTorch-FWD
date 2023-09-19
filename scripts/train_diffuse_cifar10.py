import torch
from src.improved_UNet import Improv_UNet
from src.util import _parse_args
from src.dataloader import get_dataloaders
from src.freq_math import (
    forward_wavelet_packet_transform,
    fourier_power_divergence,
    wavelet_packet_power_divergence,
)
from src.sample import sample_noise, sample_DDPM, linear_noise_scheduler
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn
import datetime
import os
from functools import partial
from tqdm import tqdm


def main():
    args = _parse_args()
    print(args)

    now = datetime.datetime.now()

    writer = SummaryWriter(log_dir=args.logdir, comment="CIFAR10")
    checkpoint_dir = os.path.join(args.logdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    # Dataloaders
    train_loader, val_loader = get_dataloaders(
        dataset_name="cifar10", batch_size=args.batch_size, data_path=None
    )
    input_shape = ()
    for data, _ in val_loader:
        input_shape = data.shape
        break
    time_steps = args.time_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using device:{device}')
    # device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    # Model instance
    if args.conditional:
        print("Note: Using class conditional.")  #
        num_classes = 1000

    if args.attn_heads_upsample == -1:
        args.attn_heads_upsample = args.attn_heads

    channel_mult = []
    if "," in args.channel_mult:
        for value in args.channel_mult.split(","):
            channel_mult.append(int(value))
    else:
        channel_mult.append(int(args.channel_mult))

    attn_res = []
    for value in args.attn_resolution.split(","):
        attn_res.append(input_shape[-1] // int(value))

    model = Improv_UNet(
        in_channels=input_shape[1],
        model_channels=args.base_channels,
        out_channels=input_shape[1],
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=tuple(attn_res),
        dropout=args.dropout,
        channel_mult=tuple(channel_mult),
        num_classes=num_classes,
        num_heads=args.attn_heads,
        num_heads_upsample=args.attn_heads_upsample,
        use_scale_shift_norm=True,
    )
    dids = [int(gpu) for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    print(dids)
    model = torch.nn.DataParallel(model, device_ids=dids)
    params = sum([p.data.nelement() for p in model.parameters()])
    print(f"Number of parameters in the model are: {params}")
    model.to(device)
    # Compile the model
    #print('Compiling the model..')
    #model = torch.compile(model)
    #torch.set_float32_matmul_precision("high")

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch}")
        model.train()
        epoch_loss = 0.0
        for i, (input, class_label) in enumerate(train_loader):
            optimizer.zero_grad()
            x, y, current_steps = load_input(input, time_steps)
            x, y, class_label = x.to(device), y.to(device), class_label.to(device)
            current_steps = current_steps.to(device)
            pred_noise = model(x, current_steps, class_label)
            mse_loss = criterion(pred_noise, y)
            mse_loss.backward()
            optimizer.step()
            epoch_loss += mse_loss.item()

            if i % 100 == 0:
                print(i, epoch_loss / (i + 1), flush=True)

        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir,f' model_{epoch}.pt'))
                print('Sampling validation set...')
                for input, class_labels in val_loader:
                    x_0 = sample_DDPM(
                        class_labels=class_labels,
                        model=model.module,
                        max_steps=time_steps,
                        input_=input,
                        device=device
                    )
                    os.makedirs('./interim_samples/', exist_ok=True)
                    torch.save(x_0, f'./interim_samples/tensor_{epoch}.pt')
                    break


def load_input(input_imgs: torch.Tensor, time_steps: int):
    current_steps = torch.randint(high=time_steps, size=[input_imgs.shape[0]])
    alphas_t = torch.tensor(
        [linear_noise_scheduler(time, time_steps)[0] for time in current_steps]
    ).reshape(len(current_steps), 1)
    batch_map = torch.vmap(sample_noise, randomness="different")
    x, y = batch_map(input_imgs, alphas_t)
    return x, y, current_steps


if __name__ == "__main__":
    main()
