import flax.linen as nn
import jax.numpy as jnp

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
        # expand dims of the emb_out - if necessary add later
        h = h + emb_out
        h = out_layers(h)

        if self.out_channels == x.shape[-1]:
            return x + h
        elif self.use_conv:
            return nn.Conv(self.out_channels, [3, 3], padding="SAME")(x) + h
        else:
            return nn.Conv(self.out_channels, [1, 1])(x) + h
