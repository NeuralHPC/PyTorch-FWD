from typing import List
from functools import partial

import pickle
from src.sample import sample_net_noise, sample_net_test, sample_noise_simple
from src.util import write_movie

import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_path = "./data/MNIST/"
    checkpoint_path = "/home/wolter/uni/diffusion/log/checkpoints/e_50_time_2023-09-04 12:34:09.926586.pkl"
    # checkpoint_path = '/home/wolter/tunnel/infgpu/uni/diffusion/log/checkpoints/e_390_time_2023-06-22 16:27:04.801459.pkl'

    with open(checkpoint_path, 'rb') as f:
        loaded = pickle.load(f)

    (net_state, opt_state, model) = loaded
    test_img = sample_net_noise(net_state, model, 4, [28, 28, 1], 40, label=2,
                                sampling_function=sample_noise_simple)
    # write_movie([s[0] for s in steps], xlim=1, ylim=1)
    plt.imshow((test_img + 1)/2.)
    plt.show()

    partial_sample_fun = partial(sample_net_test, net_state=net_state, model=model,
                                 key=2, max_steps=40, data_dir=data_path,
                                 sampling_function=sample_noise_simple)

    rec1 = partial_sample_fun(test_time_step=1)
    rec5 = partial_sample_fun(test_time_step=5)
    rec10 = partial_sample_fun(test_time_step=10)
    rec20 = partial_sample_fun(test_time_step=20)
    rec30 = partial_sample_fun(test_time_step=30)
    rec40 = partial_sample_fun(test_time_step=40)
