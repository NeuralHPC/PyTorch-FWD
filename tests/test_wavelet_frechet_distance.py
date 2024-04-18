"""Test Wavelet packet Frechet distance."""
import pytest
import torch as th
import pytest
import numpy as np

from itertools import pairwise
from src.fwd import compute_packet_statistics, compute_avg_frechet_distance
from sklearn.datasets import load_sample_images
from torchvision import transforms
from copy import deepcopy

th.set_default_dtype(th.float64)
th.use_deterministic_algorithms(True)

make_dataloader = lambda x: th.utils.data.DataLoader(th.utils.data.TensorDataset(x), batch_size=1, shuffle=False, drop_last=False)
default_params = {
    'dataloader': None,
    'wavelet': 'sym5',
    'max_level': 2,
    'log_scale': False
}

@pytest.mark.slow
def get_images(img_size: int = 64) -> th.Tensor:
    """Generate images of given size.

    Args:
        img_size (int): Integer specifying the desired image size.

    Returns:
        th.Tensor: Tensor containing images of shape [batchsize, channels, height, width].

    """
    dataset = load_sample_images()
    tower = th.Tensor(deepcopy(dataset.images[0]))
    flower = th.Tensor(deepcopy(dataset.images[1]))
    images = th.stack([tower, tower, tower, tower, flower, flower, flower, flower], axis=0)
    images = images/255.0
    images = th.permute(images, [0, 3, 1, 2])
    images = transforms.functional.resize(images, (img_size, img_size))
    assert images.shape == (8, 3, img_size, img_size)
    return images


def compute_fwd(target_images, output_images):
    default_params['dataloader'] = make_dataloader(target_images)
    mu1, sigma1 = compute_packet_statistics(**default_params)
    default_params['dataloader'] = make_dataloader(output_images)
    mu2, sigma2 = compute_packet_statistics(**default_params)
    
    distance = compute_avg_frechet_distance(mu1, mu2, sigma1, sigma2)
    return distance


@pytest.mark.slow
def test_same_input():
    target_images = get_images()
    output_images = deepcopy(target_images)
    fwd = compute_fwd(target_images, output_images)
    assert np.allclose(fwd, 0.0, atol=1e-3)


@pytest.mark.slow
def test_shuffle_input():
    target_images = get_images()
    # Shuffle the output images
    output_images = deepcopy(target_images)
    permutation = th.randperm(len(target_images))
    shuffled_images = output_images[permutation, :, :, :]
    assert not th.allclose(shuffled_images, output_images)

    shuffled_fwd = compute_fwd(target_images, shuffled_images)
    unshuffled_fwd = compute_fwd(target_images, output_images)
    assert np.allclose(shuffled_fwd, unshuffled_fwd, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("wavelet", ["sym5", "db5", "Haar"])
@pytest.mark.parametrize("level", [2, 3, 4])
def test_various_wavelets(wavelet, level):
    target_images = get_images()
    output_images = deepcopy(target_images)

    default_params['wavelet'] = wavelet
    default_params['max_level'] = level

    fwd = compute_fwd(target_images, output_images)
    assert np.allclose(fwd, 0.0, atol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("img_size_level", [(32, 1), (64, 2), (128, 3)]) #  (256, 4)
def test_various_image_sizes(img_size_level):
    size, level = img_size_level

    default_params['max_level'] = level

    target_images = get_images(size)
    output_images = deepcopy(target_images)
    assert output_images.shape == (8, 3, size, size)
    
    fwd = compute_fwd(target_images, output_images)
    assert np.allclose(fwd, 0.0, atol=1e-3)


@pytest.mark.slow
def test_checkerboard_FWD():
    import scipy.ndimage

    range_max = 100 
    # tile_size = 10 # determines tile size
    images = []

    for tile_size in [2, 5, 10]:
    # generate random grid
    
        grid = np.meshgrid(np.arange(0,range_max), np.arange(0,range_max))
        # grid = grid[0] + grid[1]

        tile_fun = lambda cord: (cord % tile_size) < (tile_size // 2)

        x_tile = tile_fun(grid[0])
        y_tile = tile_fun(grid[1])
        grid = 1 - (x_tile + y_tile)

        grid = grid.astype(np.float32)

        # generate 10 images via a random fft-shift
        img_list = []
        for _ in range(1000):
            input_ = np.fft.fft2(grid)
            shift = np.random.uniform(-1, 1)*tile_size//2
            image = scipy.ndimage.fourier_shift(input_, shift=shift)
            image = np.fft.ifft2(image)
            img_list.append(image.real)
        # B, H, W
        img_array = np.stack(img_list, axis=0)
        # B, C, H, W
        # img_array = np.expand_dims(img_array, 1)
        img_array = np.stack([img_array]*3, 1)
        images.append(img_array)

    # a list for the wavelet frecet distances.
    fwd_list = []
    reference = images[0]
    default_params['max_level'] = 3
    for cimgs in images[1:]:
        fwd = compute_fwd(target_images=th.from_numpy(reference),
                          output_images=th.from_numpy(cimgs))
        fwd_list.append(fwd)

    assert all(a < b for a, b in pairwise(fwd_list))

