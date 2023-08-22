"""UNet stated in Improved Denoising Diffusion Probabilistic Models (https://arxiv.org/abs/2102.09672)"""
from typing import Tuple

import math, jax
import jax.numpy as jnp
import flax.linen as nn
from src.Improved_UNet.utils import ResBlock, UPSample, AttentionBlock

class Improv_UNet(nn.Module):
    out_channels: int
    base_channels: int
    classes: bool
    channel_mult: Tuple[int]
    num_res_blocks: int
    num_heads: int
    num_heads_ups: int
    attention_res: Tuple[int]

    @nn.compact
    def __call__(self, x_in: Tuple[jnp.ndarray]) -> jnp.ndarray:
        x, time, label = x_in

        time_embedding = nn.Sequential([
            nn.Dense(self.base_channels * 4),
            nn.swish,
            nn.Dense(self.base_channels * 4)
        ])
        time_embeds = time_embedding(self.sinus_timestep(time))
        
        emb = time_embeds
        if self.classes:
            class_embeds = nn.Dense(self.base_channels * 4)(label)
            emb = emb + class_embeds

        def encoder_block():
            blocks =[
                nn.Conv(self.base_channels, [3, 3], padding="SAME")
            ]
            ds = 1
            for level, mult in enumerate(self.channel_mult):
                for _ in range(self.num_res_blocks):
                    blocks.append(ResBlock(out_channels=self.base_channels * mult))
                    if ds in self.attention_res:
                        blocks.append(
                            AttentionBlock(
                            channels=self.base_channels * mult,
                            num_heads=self.num_heads
                            )
                        )
                if level != len(self.channel_mult) - 1:
                    # Downsample
                    blocks.append(
                        nn.Conv(mult*self.base_channels, [3, 3], strides=2, padding="SAME")
                    )
                    ds *= 2
            return blocks, ds

        def middle_block():
            return [
                ResBlock(out_channels=self.base_channels*self.channel_mult[-1]),
                AttentionBlock(channels=self.base_channels*self.channel_mult[-1],
                               num_heads=self.num_heads),
                ResBlock(out_channels=self.base_channels*self.channel_mult[-1]),
            ]

        def decoder_block(ds):
            blocks = []
            for level, mult in list(enumerate(self.channel_mult))[::-1]:
                for i in range(self.num_res_blocks+1):
                    blocks.append(ResBlock(out_channels=self.base_channels*mult, use_conv=True))
                    if ds in self.attention_res:
                        blocks.append(
                            AttentionBlock(
                            channels=self.base_channels * mult,
                            num_heads=self.num_heads_ups
                            )
                        )
                    if level and i == self.num_res_blocks:
                        ds //= 2
                        blocks.append(UPSample(out_channels=self.base_channels*mult))
            return blocks

        hs = []
        h = x
        enc_block, ds = encoder_block()

        for block in enc_block:
            if isinstance(block, ResBlock):
                h = block(h, emb)
            else:
                h = block(h)
            if not isinstance(block, AttentionBlock):
                hs.append(h)

        for block in middle_block():
            if isinstance(block, ResBlock):
                h = block(h, emb)
            else:
                h = block(h)

        for block in decoder_block(ds):
            if isinstance(block, ResBlock):
                p = hs.pop()
                cat_in = jnp.concatenate([h, p], axis=-1)
                h = block(cat_in, emb)
            else:
                h = block(h)

        final_conv = nn.Sequential([
            nn.GroupNorm(),
            nn.swish,
            nn.Conv(self.out_channels, [3, 3], padding="SAME")
        ])

        return final_conv(h)


    def sinus_timestep(self, timesteps, max_period=10000):
        half = self.base_channels// 2
        freqs = jnp.exp(
            -math.log(max_period) * jnp.arange(0, half) / half
        )
        args = timesteps[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if self.base_channels%2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding
