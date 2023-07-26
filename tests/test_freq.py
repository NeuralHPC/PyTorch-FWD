from itertools import product

import scipy
import jax.numpy as jnp

from src.freq_math import (
    get_freq_order,
    generate_frequency_packet_image,
    process_images,
    inverse_wavelet_packet_transform
)

def test_loss():
    import scipy.datasets
    import matplotlib.pyplot as plt
    face = jnp.array(scipy.datasets.face())
    face = jnp.stack([face, face, face, face], axis=0)
    face = face.astype(jnp.float64)/255.

    _, natural_path = get_freq_order(level=3)
    packets = process_images(face, natural_path, wavelet="Haar")
    p_image = generate_frequency_packet_image(packets, 3)
    assert p_image.shape == (4, 768, 1024, 3)

def test_inverse_wp():
    face = jnp.array(scipy.datasets.face())
    face = jnp.stack([face, face, face, face], axis=0)
    face = face.astype(jnp.float64)/255.
    
    natural_path = list(product(["a", "h", "v", "d"], repeat=3))
    packets = process_images(face, natural_path, max_level = 3, wavelet = "db3")

    reconstruction = inverse_wavelet_packet_transform(packets, max_level=3, wavelet="db3")

    assert jnp.max(jnp.abs(reconstruction[:, :768, :1024] - face)) < 1e-5
