from typing import List, Union, Tuple

import pickle
import jax
import random
import sys, os
import jax.numpy as jnp
import numpy as np

from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
from src.sample import sample_DDPM, batch_DDPM
from src.util import write_movie, _sampler_args, get_label_dict
from functools import partial
from tqdm import tqdm
from src.fid import inception, fid
import matplotlib.pyplot as plt
import time


def sample_30K(args, net_state, model, labels):
    batch_size = 15000 # Reduce this number in case of low memory
    print(batch_size)
    base_path = "sample_imgs"
    os.makedirs(base_path, exist_ok=True)

    # inception_model = inception.InceptionV3(pretrained=True)
    # inception_net_state = inception_model.init(
    #     jax.random.PRNGKey(0),
    #     jnp.ones((1, 256, 256, 3))
    # )
    # apply_fn = jax.jit(partial(inception_model.apply, train=False))

    gpus = args.gpus if args.gpus > 0 else jax.local_device_count()
    # Sample all the images
    #jit_batch_DDPM = jax.jit(batch_DDPM, static_argnames=['net_state', 'model', 'key', 'input_shape', 
    #                                                      'max_steps', 'batch_size'])
    sample_partial = jax.pmap(partial(batch_DDPM, 
                              net_state=net_state,
                              key = jax.random.PRNGKey(args.seed),
                              input_shape=[args.input_shape, args.input_shape, 3],
                              max_steps=args.diff_steps,
                              batch_size=batch_size//gpus),
                              devices=jax.devices()[:gpus],
                             )

    # sample_partial = jax.vmap(partial(batch_DDPM, 
    #                             net_state=net_state, model=model,
    #                             key=args.seed, input_shape=[args.input_shape, args.input_shape, 3],
    #                             max_steps=args.diff_steps, batch_size=batch_size))

    def rescale(img):
        img = (img - jnp.min(img))/(jnp.max(img) - jnp.min(img))*255.0
        return img.astype(jnp.uint8)
    
    # def save_fig(imgs, count):
    #     for num, img in enumerate(imgs):
    #         plt.imshow(img)
    #         plt.axis("off")
    #         nmbr = count + num
    #         fnm = os.path.join(base_path, f"{nmbr:05d}.jpg")
    #         plt.savefig(fnm, dpi=600, bbox_inches="tight")
    #         plt.close()

    # batch_activations = []
    count = 0
    for idx in range(len(labels)//batch_size):
        print(f"Processing batch: {idx+1}/{len(labels)//batch_size}")
        lbls = labels[idx*batch_size : (idx+1)*batch_size]
        lbl = jnp.expand_dims(jnp.stack(jnp.split(lbls, gpus)), -1)
        start = time.time()
        sampled_imgs = sample_partial(test_label=lbl)
        print(f"Processing time: {time.time()-start}")
        sampled_imgs = jnp.reshape(sampled_imgs, (batch_size, args.input_shape, args.input_shape, 3))
        # sampled_imgs = jax.vmap(rescale)(sampled_imgs)
        fnm = os.path.join(base_path, f"sampled_imgs_{count}.npz")
        jnp.savez(fnm, imgs=sampled_imgs)
        count += 1
        # acts = fid.compute_sampled_statistics(sampled_imgs,
        #                                     inception_net_state,
        #                                     apply_fn)
        # batch_activations.append(acts)
    # batch_activations = jnp.concatenate(batch_activations, axis=0)
    # Compute mean and variance
    # mu = np.mean(batch_activations, axis=0)
    # sigma = np.cov(batch_activations, rowvar=False)

    # root_save_path="sample_imgs"
    # os.makedirs(root_save_path, exist_ok=True)
    # fname = os.path.join(root_save_path, f"stats_{args.input_shape}x{args.input_shape}.npz")
    # np.savez(fname, mu=mu, sigma=sigma)
    # print(f"Sampling complete and precomputed mean and sigma for FID computation are saved at {fname}")


if __name__ == "__main__":
    args = _sampler_args()
    with open(args.ckpt_path, "rb") as fp:
        loaded = pickle.load(fp)
   
    labels_dict = get_label_dict("data/CelebAHQ/identity_CelebAHQ.txt")
    labels = jnp.array(list(labels_dict.values()))

    (net_state, opt_state, model) = loaded

    # Sample 1 image and save as GIF
    if args.gif:
        seed = random.randint(0, 10000)
        key = jax.random.PRNGKey(seed)
        
        label = jax.random.choice(key, labels, [1], replace=False)
        print(seed,label)

        test_img, steps = sample_DDPM(net_state, model, args.seed,
                                      [args.input_shape, args.input_shape, 3],
                                      args.diff_steps, label[0])
        
        plt.imshow(test_img)
        plt.axis("off")
        plt.savefig("final_img.png", dpi=600, bbox_inches="tight")
        plt.close()

        write_movie([s[0] for s in steps], xlim=1, ylim=1)
        sys.exit("Writing complete")
    
    sample_30K(args, net_state, model, labels)