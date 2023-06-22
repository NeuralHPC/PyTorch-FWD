import pickle
import jax
import jax.numpy as jnp

from src.sample import sample_net_noise, sample_net_test


import matplotlib.pyplot as plt

if __name__ == '__main__':
    checkpoint_path = "/home/wolter/uni/diffusion/log/checkpoints/e_5_time_2023-06-22 12:35:07.968094.pkl"

    with open(checkpoint_path, 'rb') as f:
        loaded = pickle.load(f)

    (net_state, opt_state, model) = loaded
    #test_img = sample_net_noise(net_state, model, 4, [28, 28], 40)
    #plt.imshow((test_img + 1)/2.)
    #plt.show()


    rec1 = sample_net_test(net_state, model, 2, 1)
    rec5 = sample_net_test(net_state, model, 2, 5)
    rec10 = sample_net_test(net_state, model, 2, 10)
    rec20 = sample_net_test(net_state, model, 2, 20)
    rec30 = sample_net_test(net_state, model, 2, 30)
    rec40 = sample_net_test(net_state, model, 2, 40)
    pass