import argparse
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
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
    

def main():
    global args
    ref_path = args.ref_path
    sample_path = args.sample_path
    batch_size = 100

    level_dict = {
        32 : 1,
        64 : 2,
        128: 3,
        256: 4,
    }

    refloader = DataLoader(ImgSet(data_path=ref_path), batch_size=batch_size, shuffle=True, pin_memory=True)
    sampleloader = DataLoader(ImgSet(data_path=sample_path), batch_size=batch_size, shuffle=True, pin_memory=True)

    fft_vals, fft_vals_inv = 0.0, 0.0
    packet_vals, packet_vals_inv = 0.0, 0.0

    print(f"Number of reference batches: {len(refloader)}")
    print(f"Number of sample batches: {len(sampleloader)}")

    min_loader = refloader
    if len(refloader) > len(sampleloader):
        min_loader = sampleloader

    # for _ in tqdm(range(len(min_loader))):
    #     ref_batch = next(iter(refloader))
    #     sample_batch = next(iter(sampleloader))
    #     a, b = fourier_power_divergence(ref_batch, sample_batch)
    #     fft_vals += a
    #     fft_vals_inv += b
    #     x, y = wavelet_packet_power_divergence(ref_batch, sample_batch, level=level)
    #     packet_vals += x
    #     packet_vals_inv += y
    ref_imgs, sample_imgs = [], []
    for _ in tqdm(range(len(min_loader))):
        ref_imgs.append(next(iter(refloader)))
        sample_imgs.append(next(iter(sampleloader)))
    ref_imgs = torch.cat(ref_imgs, axis=0)
    sample_imgs = torch.cat(sample_imgs, axis=0)
    
    fft_vals, fft_vals_inv = fourier_power_divergence(sample_imgs, ref_imgs)
    print("wavelet level: ", int(level_dict[int(sample_imgs.shape[-2])]))
    packet_vals, packet_vals_inv = wavelet_packet_power_divergence(sample_imgs, ref_imgs, level=int(level_dict[int(sample_imgs.shape[-2])]))

    print(f"PSKL FFT A->B: {round(fft_vals.item(),5)}, B->A: {round(fft_vals_inv.item(),5)}")
    print(f"PSKL Packet A->B: {round(packet_vals.item(),5)}, B->A: {round(packet_vals_inv.item(),5)}")
    

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
