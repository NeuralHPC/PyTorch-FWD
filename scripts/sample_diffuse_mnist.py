import pickle

from src.sample import sample_net
from src.networks import UNet

from src.util import get_mnist_test_data

import matplotlib.pyplot as plt

if __name__ == '__main__':
    checkpoint_path = "/home/wolter/tunnel/bender/uni/diffusion/log/checkpoints/2023-06-19 15:19:24.543303"

    with open(checkpoint_path, 'rb') as f:
        loaded = pickle.load(f)

    model = UNet()

    (net_state, opt_state) = loaded
    test_img = sample_net(net_state, model, 4, [28, 28], 30)
    plt.imshow(test_img)
    plt.show()

    pass 