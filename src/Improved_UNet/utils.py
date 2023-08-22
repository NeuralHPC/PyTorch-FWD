from typing import Any
import flax.linen as nn
import jax.numpy as jnp
import jax
import math

class ResBlock(nn.Module):
    out_channels: int
    use_conv: bool = False
    use_scale_shift_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: jnp.ndarray) -> jnp.ndarray:
        in_layers = nn.Sequential([
            nn.GroupNorm(),
            nn.swish,
            nn.Conv(self.out_channels, [3, 3], padding='SAME')
        ])
        emd_layers = nn.Sequential([
            nn.swish,
            nn.Dense(2 * self.out_channels if self.use_scale_shift_norm else self.out_channels) # is scale_shift_norm necesarry?
        ])
        out_layers = [
            nn.GroupNorm(),
            nn.swish,
            # Dropped the dropout layer here - if necessary add later
            nn.Conv(self.out_channels, [3, 3], padding="SAME") # Torch has these params zeroed out and detached - if necessary add later
        ]

        h = in_layers(x)
        emb_out = emd_layers(emb)
        emb_out = jnp.expand_dims(emb_out, axis=[1, 2])
        if self.use_scale_shift_norm:
            out_norm, out_rest = out_layers[0], nn.Sequential(out_layers[1:])
            scale, shift = jnp.split(emb_out, 2, axis=-1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            out_layers = nn.Sequential(out_layers)
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
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = x.shape
        x = jax.image.resize(x, (B, H * 2, W * 2, C), 'nearest')
        return nn.Conv(self.out_channels, [3, 3], padding="SAME")(x)
    

class AttentionBlock(nn.Module):
    channels: int
    num_heads: int = 1

    @nn.compact
    def __call__(self, x) -> Any:
        B, H, W, C = x.shape
        # x = jnp.reshape(x, (B, C, -1))
        x = nn.GroupNorm()(x)
        qkv = nn.Conv(self.channels * 3, [1], padding="SAME")(x)
        qkv = jnp.reshape(qkv, (B, qkv.shape[-1], -1))
        qkv = jnp.reshape(qkv, (B * self.num_heads, -1, qkv.shape[2]))
        h = self.attention(qkv)
        h = jnp.reshape(h, (B, h.shape[-1], -1))
        h = nn.Conv(self.channels, [1])(h).reshape(B, H, W, C)
        return x + h

    def attention(self, qkv: jnp.ndarray) -> jnp.ndarray:
        ch = qkv.shape[1] // 3
        q, k ,v = jnp.split(qkv, 3, axis=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = jnp.einsum(
            "bct, bcs -> bts", q*scale, k*scale
        )
        weight = nn.softmax(weight, axis=-1)
        return jnp.einsum("bts, bcs -> bct", weight, v)
