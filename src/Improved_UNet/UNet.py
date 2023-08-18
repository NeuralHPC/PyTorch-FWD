"""UNet stated in Improved Denoising Diffusion Probabilistic Models (https://arxiv.org/abs/2102.09672)"""
from typing import Tuple

import math
import jax.numpy as jnp
import flax.linen as nn
from src.Improved_UNet.utils import ResBlock, UPSample

class Improv_UNet(nn.Module):
    out_channels: int
    model_channels: int
    classes: int
    channel_mult = [1, 2, 2, 4] # 64x64 TODO: Make this a CLA
    num_res_blocks: int = 2 # 64x64 TODO: Make this a CLA

    @nn.compact
    def __call__(self, x_in: Tuple[jnp.ndarray]) -> jnp.ndarray:
        x, time, label = x_in

        time_embedding = nn.Sequential([
            nn.Dense(self.model_channels * 4),
            nn.silu,
            nn.Dense(self.model_channels * 4)
        ])
        time_embeds = time_embedding(self.sinus_timestep(time, self.model_channels))
        
        emb = time_embeds
        if self.classes is not None:
            class_embeds = nn.Dense(self.model_channels * 4)(label)
            emb = emb + class_embeds

        def encoder_block():
            blocks =[
                nn.Conv(self.model_channels, [3, 3], padding="SAME")
            ]
            for level, mult in enumerate(self.channel_mult):
                for _ in range(self.num_res_blocks):
                    blocks.append(ResBlock(out_channels=self.model_channels * mult))

                if level != len(self.channel_mult) - 1:
                    # Downsample
                    blocks.append(
                        nn.Conv(mult*self.model_channels, [3, 3], strides=2, padding="SAME")
                    )
            return blocks

        def middle_block():
            return [
                ResBlock(out_channels=self.model_channels*self.channel_mult[-1]),
                ResBlock(out_channels=self.model_channels*self.channel_mult[-1])
            ]

        def decoder_block():
            blocks = []
            for level, mult in list(enumerate(self.channel_mult))[::-1]:
                for i in range(self.num_res_blocks+1):
                    blocks.append(ResBlock(out_channels=self.model_channels*mult, use_conv=True))
                    if level and i == self.num_res_blocks:
                        blocks.append(UPSample(out_channels=self.model_channels*mult))
            return blocks

        hs = []
        h = x
        for block in encoder_block():
            if isinstance(block, ResBlock):
                h = block(h, emb)
            else:
                h = block(h)
            hs.append(h)

        for block in middle_block():
            h = block(h, emb)

        for block in decoder_block():
            if isinstance(block, ResBlock):
                cat_in = jnp.concatenate([h,hs.pop()], axis=-1)
                h = block(cat_in, emb)
            else:
                h = block(h)

        
        final_conv = nn.Sequential([
            nn.GroupNorm(),
            nn.silu,
            nn.Conv(self.out_channels, [3, 3], padding="SAME")
        ])

        return final_conv(h)


    def sinus_timestep(self, timesteps, dim , max_period=10000):
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period) * jnp.arange(0, half) / half
        )
        args = timesteps[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim%2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding
