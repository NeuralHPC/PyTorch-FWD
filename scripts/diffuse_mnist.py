import numpy as np
import jax.numpy as jnp

from datasets import load_dataset


if __name__ == '__main__':
    batch_size = 64
    dataset = load_dataset("mnist")
    train_batches = np.array_split(
        [np.array(img) for img in dataset['train']['image']],
        len(dataset['train']['image']) // batch_size)
    breakpoint()
    for img in train_batches:
        pass
