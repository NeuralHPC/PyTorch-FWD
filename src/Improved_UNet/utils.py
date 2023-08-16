from typing import Any
import flax.linen as nn
import jax.numpy as jnp
import jax

class ResBlock(nn.Module):
    out_channels: int
    use_conv: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: jnp.ndarray) -> jnp.ndarray:
        in_layers = nn.Sequential([
            nn.GroupNorm(),
            nn.silu,
            nn.Conv(self.out_channels, [3, 3], padding='SAME')
        ])
        emd_layers = nn.Sequential([
            nn.silu,
            nn.Dense(self.out_channels) # is scale_shift_norm necesarry?
        ])
        out_layers = nn.Sequential([
            nn.GroupNorm(),
            nn.silu,
            # Dropped the dropout layer here - if necessary add later
            nn.Conv(self.out_channels, [3, 3], padding="SAME") # Torch has these params zeroed out and detached - if necessary add later
        ])

        h = in_layers(x)
        emb_out = emd_layers(emb)
        emb_out = jnp.expand_dims(emb_out, axis=[1, 2])
        h = h + emb_out
        h = out_layers(h)

        if self.out_channels == x.shape[-1]:
            return x + h
        elif self.use_conv:
            return nn.Conv(self.out_channels, [3, 3], padding="SAME")(x) + h
        else:
            return nn.Conv(self.out_channels, [1, 1])(x) + h

class UPSample(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: jnp.array=None) -> jnp.ndarray:
        B, H, W, C = x.shape
        x = jax.image.resize(x, (B, H * 2, W * 2, C), 'nearest')
        return nn.Conv(self.out_channels, [3, 3], padding="SAME")(x)