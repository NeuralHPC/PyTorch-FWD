import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

# Load all the images
files = glob.glob('./sample_imgs/*.npz')
images = []
for file in files:
    with jnp.load(file) as data:
        images.append(data['imgs'])
images = jnp.concatenate(images, axis=0)

# Get random indices for plot
rand_indices = np.random.choice(range(len(images)), 100, replace=False)

plot_img = []
nos = 8
# Concatenate images to form a grid
for i in range(nos):
    row_img = images[rand_indices[i*nos: (i+1)*nos], :, :, :]
    row_img = jnp.concatenate(row_img, axis=-2)
    plot_img.append(row_img)
plot_img = jnp.concatenate(plot_img, axis=-3)

# Plot the image
plt.figure(figsize=(20, 20))
plt.imshow(plot_img)
plt.axis('off')
plt.savefig(f"./Img_{nos}x{nos}.png", dpi=600, bbox_inches='tight')
plt.close()