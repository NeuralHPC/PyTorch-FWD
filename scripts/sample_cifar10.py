import torch
import os
from src.sample import sample_DDPM
from src.improved_UNet import Improv_UNet
from src.dataloader import get_dataloaders
from config.cifar10 import dataset

sample_dir = "./sample_imgs"
os.makedirs(sample_dir, exist_ok=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train_loader, val_loader = get_dataloaders("cifar10", 2500, 2500, ".")

model = Improv_UNet(
    in_channels=3,
    model_channels=128,
    out_channels=3,
    num_res_blocks=2,
    attention_resolutions=tuple([2]),
    dropout=0.1,
    channel_mult=tuple([1, 2, 2, 2]),
    num_classes=1000,
    num_heads=4,
    num_heads_upsample=4,
    use_scale_shift_norm=True,
)

model = model.to(device)
model.load_state_dict(torch.load("./log/checkpoints/model_1000.pt"))
with torch.no_grad():
    model.eval()
    imgs = []
    count = 0
    std = torch.reshape(torch.tensor(dataset["std"]), (1, 3, 1, 1)).to(device)
    mean = torch.reshape(torch.tensor(dataset["mean"]), (1, 3, 1, 1)).to(device)
    for input_, class_labels in train_loader:
        x_0 = sample_DDPM(
            class_labels=class_labels,
            model=model,
            max_steps=1000,
            input_=input_,
            device=device,
        )
        count += 1
        x_0 = (x_0 * std) + mean
        print(x_0.shape, count)
        imgs.append(x_0)
    img = torch.concat(imgs, axis=0)
    torch.save(img, os.path.join(sample_dir, "sampled_tensors.pt"))
