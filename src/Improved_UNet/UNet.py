"""UNet stated in Improved Denoising Diffusion Probabilistic Models (https://arxiv.org/abs/2102.09672)"""
from typing import Tuple

import math, jax
import jax.numpy as jnp
import flax.linen as nn
from src.Improved_UNet.utils import ResBlock, UPSample, AttentionBlock

class Improv_UNet(nn.Module):
    out_channels: int
    model_channels: int
    classes: int
    channel_mult = [1, 1, 2, 2, 4] # 64x64 TODO: Make this a CLA
    num_res_blocks: int = 2 # 64x64 TODO: Make this a CLA
    num_heads: int = 1
    num_heads_ups: int = 1
    attention_res = [8, 16]

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
            ds = 1
            for level, mult in enumerate(self.channel_mult):
                for _ in range(self.num_res_blocks):
                    blocks.append(ResBlock(out_channels=self.model_channels * mult))
                    if ds in self.attention_res:
                        blocks.append(
                            AttentionBlock(
                            channels=self.model_channels * mult,
                            num_heads=self.num_heads
                            )
                        )
                if level != len(self.channel_mult) - 1:
                    # Downsample
                    blocks.append(
                        nn.Conv(mult*self.model_channels, [3, 3], strides=2, padding="SAME")
                    )
                    ds *= 2
            return blocks, ds

        def middle_block():
            return [
                ResBlock(out_channels=self.model_channels*self.channel_mult[-1]),
                AttentionBlock(channels=self.model_channels*self.channel_mult[-1],
                               num_heads=self.num_heads),
                ResBlock(out_channels=self.model_channels*self.channel_mult[-1]),
            ]

        def decoder_block(ds):
            blocks = []
            for level, mult in list(enumerate(self.channel_mult))[::-1]:
                for i in range(self.num_res_blocks+1):
                    blocks.append(ResBlock(out_channels=self.model_channels*mult, use_conv=True))
                    if ds in self.attention_res:
                        blocks.append(
                            AttentionBlock(
                            channels=self.model_channels * mult,
                            num_heads=self.num_heads_ups
                            )
                        )
                    if level and i == self.num_res_blocks:
                        ds //= 2
                        blocks.append(UPSample(out_channels=self.model_channels*mult))
            return blocks

        hs = []
        h = x
        enc_block, ds = encoder_block()
        print(h.shape)
        for block in enc_block:
            if isinstance(block, ResBlock):
                h = block(h, emb)
            else:
                h = block(h)
            print(h.shape, type(block))
            hs.append(h)
        print(len(hs))
        for block in middle_block():
            if isinstance(block, ResBlock):
                h = block(h, emb)
            else:
                h = block(h)
            print(h.shape, type(block))

        print('Decoder block')
        for block in decoder_block(ds):
            print(h.shape, type(block))
            if isinstance(block, ResBlock):
                p = hs.pop()
                cat_in = jnp.concatenate([h, p], axis=-1)
                h = block(cat_in, emb)
            elif isinstance(block, AttentionBlock):
                #cat_in = jnp.concatenate([h, p], axis=-1)
                h = block(h)
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

if __name__ == '__main__':
    input_shape = [64, 64, 3]
    model = Improv_UNet(
        out_channels=input_shape[-1],
        model_channels=128,
        classes=1000
    )
    key = jax.random.PRNGKey(42)
    batch_size = 10
    net_state = model.init(
        key,
        (jnp.ones([batch_size] + input_shape),
             jnp.ones([batch_size]),
             jnp.expand_dims(jnp.ones([batch_size]), -1))
    )
    dummy_input = (jnp.zeros([batch_size] + input_shape),
             jnp.ones([batch_size]),
             jnp.expand_dims(jnp.ones([batch_size]), -1))
    dummy_output = model.apply(net_state, dummy_input)
    assert dummy_output.shape == tuple([batch_size]+input_shape)