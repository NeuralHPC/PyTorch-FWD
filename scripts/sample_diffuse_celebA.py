from typing import List, Union, Tuple

import pickle
import jax
import random
import sys, os
import jax.numpy as jnp
import numpy as np

from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
from src.sample import sample_noise
from src.util import write_movie, _sampler_args, get_label_dict
from functools import partial
from tqdm import tqdm


def sample_net_noise(net_state: FrozenDict, model: nn.Module, key: int,
                    input_shape: List[int], max_steps: int, test_label: int,
                    is_gif: bool) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    """Sample images from noise.

    Args:
        net_state (FrozenDict): Model state
        model (nn.Module): Model object
        key (int): Seed values
        input_shape (List[int]): Input shape
        max_steps (int): Number of diffusion steps
        test_label (int): Test labels for conditioning
        is_gif (bool): Boolean for saving GIF

    Returns:
        Union[np.ndarray, Tuple(np.ndarray, List): Either returns sampled image or 
                    whole sample process along with sampled image
    """
    if key == -1:
        key = random.randint(0, 50000)
    prng_key = jax.random.PRNGKey(key)
    process_array = jax.random.normal(
        prng_key, shape=[1] + input_shape
    )
    if is_gif:
        steps = [process_array]

    for time in reversed(range(max_steps)):
        denoise = model.apply(net_state,
                              (process_array,
                               jnp.expand_dims(jnp.array(time), -1),
                               jnp.expand_dims(jnp.array(test_label), 0)))
        process_array += denoise
        prng_key = jax.random.split(prng_key, 1)[0]
        if is_gif:
            steps.append(process_array)
        process_array = sample_noise(process_array, time, prng_key, max_steps)[0]
    
    if is_gif:
        return process_array[0], steps
    return process_array[0]


if __name__ == "__main__":
    args = _sampler_args()
    batch_size = 1000 # Reduce this number in case of low memory
    with open(args.ckpt_path, "rb") as fp:
        loaded = pickle.load(fp)
   
    labels_dict = get_label_dict("data/CelebAHQ/identity_CelebAHQ.txt")
    labels = jnp.array(list(labels_dict.values()))

    (net_state, opt_state, model) = loaded
    
    # Sample 1 image and save as GIF
    if args.gif:
        key = jax.random.PRNGKey(random.randint(0, 10000))
        label = jax.random.choice(key, labels, [1], replace=False)
        test_img, steps = sample_net_noise(net_state, model, args.seed, [args.input_shape, args.input_shape, 3], args.diff_steps, label[0], args.gif)
        write_movie([s[0] for s in steps], xlim=1, ylim=1)
        sys.exit("Writing complete")
    
    # Sample all the images
    sample_partial = jax.vmap(partial(sample_net_noise, 
                             net_state=net_state, model=model,
                             key=args.seed, input_shape=[args.input_shape, args.input_shape, 3],
                             max_steps=args.diff_steps, is_gif=args.gif))
    for idx in tqdm(range(len(labels)//batch_size)):
        sampled_imgs = sample_partial(test_label=labels[idx*batch_size:(idx+1)*batch_size])
        root_save_path="sample_imgs"
        os.makedirs(root_save_path, exist_ok=True)
        file_name = os.path.join(root_save_path,f"sample_{args.input_shape}x{args.input_shape}_{idx+1}.npy")
        with open(file_name, 'wb') as fp:
            jnp.save(fp, sampled_imgs)
    print("Sampling complete")