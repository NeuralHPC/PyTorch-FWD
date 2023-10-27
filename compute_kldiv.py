import argparse
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from PIL import Image
import numpy as np
from src.freq_math import fourier_power_divergence, wavelet_packet_power_divergence

class ImgSet(Dataset):
    def __init__(self, data_path: str) -> None:
        super().__init__()
        self.files = glob.glob(f"{data_path}*.png")
        if len(self.files) == 0:
            print("No png found, so trying for JPEG")
            self.files = glob.glob(f"{data_path}*.jpg")
        if len(self.files) == 0:
            raise ValueError("Improper datapath")

        self.transforms = transforms.Compose([
            transforms.PILToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index) -> torch.Tensor:
        img = Image.open(self.files[index])
        img = self.transforms(img)/127.5 - 1
        return img
    

class TensorSet(Dataset):
    def __init__(self, data: torch.Tensor) -> None:
        super().__init__()
        self.data = torch.from_numpy(data)/127.5 - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> torch.Tensor:
        img = self.data[index, :, :, :]
        return img.permute(2, 0, 1)


def main():
    global args
    ref_path = args.ref_path
    sample_path = args.sample_path
    batch_size = 100
    lsun_dataset = False

    level_dict = {
        32 : 1,
        64 : 2,
        128: 3,
        256: 4,
    }
    if 'imagenet' in ref_path.lower():
        data = np.load(ref_path)['arr_0']
        refloader = DataLoader(TensorSet(data=data), batch_size=batch_size, shuffle=True, pin_memory=True)
    elif ('church' not in ref_path.lower()) and ('bedroom' not in ref_path.lower()):
        refloader = DataLoader(ImgSet(data_path=ref_path), batch_size=batch_size, shuffle=True, pin_memory=True)
    else:
        lsun_dataset = True
        cls ='church_outdoor_train' if 'church' in ref_path else 'bedroom_train'
        transfs = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
        dataset = datasets.LSUN(ref_path, classes=[cls], transform=transfs)
        refloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    sampleloader = DataLoader(ImgSet(data_path=sample_path), batch_size=batch_size, shuffle=True, pin_memory=True)

    fft_vals, fft_vals_inv = 0.0, 0.0
    packet_vals, packet_vals_inv = 0.0, 0.0

    print(f"Number of reference batches: {len(refloader)}")
    print(f"Number of sample batches: {len(sampleloader)}")

    min_loader = refloader
    if len(refloader) > len(sampleloader):
        min_loader = sampleloader

    ref_imgs, sample_imgs = [], []
    no_imgs = 0
    for _ in tqdm(range(len(min_loader))):
        if lsun_dataset:
            ref_imgs.append(next(iter(refloader))[0])
            if no_imgs >= 30000:
                break
        else:
            ref_imgs.append(next(iter(refloader)))
        sample_imgs.append(next(iter(sampleloader)))
        no_imgs += batch_size

    ref_imgs = torch.cat(ref_imgs, axis=0)
    sample_imgs = torch.cat(sample_imgs, axis=0)
    
    fft_vals, fft_vals_inv = fourier_power_divergence(sample_imgs, ref_imgs)
    print("wavelet level: ", int(level_dict[int(sample_imgs.shape[-2])]))
    packet_vals, packet_vals_inv, fpd = wavelet_packet_power_divergence(sample_imgs, ref_imgs, level=int(level_dict[int(sample_imgs.shape[-2])]))
    fft_mean = 0.5*(fft_vals + fft_vals_inv)
    packet_mean = 0.5*(packet_vals + packet_vals_inv)

    print(f"PSKL FFT A->B: {round(fft_vals.item(),2)}, B->A: {round(fft_vals_inv.item(),2)}, Mean: {round(fft_mean.item(), 2)}")
    print(f"PSKL Packet A->B: {round(packet_vals.item(),2)}, B->A: {round(packet_vals_inv.item(),2)}, Mean: {round(packet_mean.item(), 2)}")
    print(f"Frechet Packet Distance: {fpd}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Dummy', description='Dummy'
    )
    parser.add_argument(
        "--ref-path",
        type=str,
        required=True,
        help="Reference path"
    )
    parser.add_argument(
        "--sample-path",
        type=str,
        required=True,
        help="Sample path"
    )
    args = parser.parse_args()
    main()
