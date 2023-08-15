from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from utils import ResBlock

class Improv_UNet(nn.Module):
    out_channels: int
    model_channels: int
    classes: int

    @nn.compact
    def __call__(self, x_in: Tuple[jnp.ndarray]) -> jnp.ndarray:
        def downsample(x, channels):
            return nn.Conv(channels, [3, 3], strides=2, padding="SAME")(x)
        
        def upsample(x, channels):
            B, H, W, C = x.shape
            x = jax.image.resize(x, (B, H * 2, W * 2, C), 'nearest')
            return nn.Conv(channels, [3, 3], padding="SAME")(x)

        x, time, label = x_in

        time_embedding = nn.Sequential([
            nn.Dense(self.model_channels * 4),
            nn.silu,
            nn.Dense(self.model_channels * 4)
        ])
        time_embeds = time_embedding(time)
        
        emb = time_embeds
        if self.classes is not None:
            class_embeds = nn.Dense(self.model_channels * 4)(nn.one_hot(label, num_classes=self.classes))
            emb = emb + class_embeds

        # Downsampling blocks
        dp1 = nn.Conv(self.model_channels, [3, 3], padding="SAME")(x)
        dp2 = ResBlock(
            out_channels=self.model_channels,
        )(dp1, emb)
        dp3 = ResBlock(
            out_channels=self.model_channels,
        )(dp2, emb)
        
        dps1 = downsample(dp3, self.model_channels)
        
        dp4 = ResBlock(
            out_channels=self.model_channels * 2,
        )(dps1, emb)
        dp5 = ResBlock(
            out_channels=self.model_channels * 2,
        )(dp4, emb)

        dps2 = downsample(dp5, self.model_channels * 2)

        dp6 = ResBlock(
            out_channels=self.model_channels * 3,
        )(dps2, emb)
        dp7 = ResBlock(
            out_channels=self.model_channels * 3,
        )(dp6, emb)

        dps3 = downsample(dp7, self.model_channels * 3)
        
        dp8 = ResBlock(
            out_channels=self.model_channels * 4,
        )(dps3, emb)
        dp9 = ResBlock(
            out_channels=self.model_channels * 4,
        )(dp8, emb)

        # Middle blocks
        mp1 = ResBlock(
            out_channels=self.model_channels * 4,
        )(dp9, emb)
        mp2 = ResBlock(
            out_channels=self.model_channels * 4,
        )(mp1, emb)

        # Upsampling blocks
        up1 = ResBlock(
            out_channels=self.model_channels * 4,
            use_conv=True
        )(mp2, emb)
        up1 = up1 + dp9
        up2 = ResBlock(
            out_channels=self.model_channels * 4,
            use_conv=True
        )(up1, emb)
        up2 = up2 + dp8
        up3 = ResBlock(
            out_channels=self.model_channels * 4,
            use_conv=True
        )(up2, emb)
        up3 = up3 + dps3

        ups1 = upsample(up3, self.model_channels * 4)

        up4 = ResBlock(
            out_channels=self.model_channels * 3,
            use_conv=True
        )(ups1, emb)
        up4 = up4 + dp7
        up5 = ResBlock(
            out_channels=self.model_channels * 3,
            use_conv=True
        )(up4, emb)
        up5 = up5 + dp6
        up6 = ResBlock(
            out_channels=self.model_channels * 3,
            use_conv=True
        )(up5, emb)
        up6 = up6 + dps2

        ups2 = upsample(up6, self.model_channels * 3)

        up7 = ResBlock(
            out_channels=self.model_channels * 2,
            use_conv=True
        )(ups2, emb)
        up7 = up7 + dp5
        up8 = ResBlock(
            out_channels=self.model_channels * 2,
            use_conv=True
        )(up7, emb)
        up8 = up8 + dp4
        up9 = ResBlock(
            out_channels=self.model_channels * 2,
            use_conv=True
        )(up8, emb)
        up9 = up9 + dps1
        
        ups3 = upsample(up9, self.model_channels * 2)

        up10 = ResBlock(
            out_channels=self.model_channels,
            use_conv=True
        )(ups3, emb)
        up10 = up10 + dp3
        up11 = ResBlock(
            out_channels=self.model_channels,
            use_conv=True
        )(up10, emb)
        up11 = up11 + dp2
        up12 = ResBlock(
            out_channels=self.model_channels,
            use_conv=True
        )(up11, emb)
        up12 = up12 + dp1

        final_conv = nn.Sequential([
            nn.GroupNorm(),
            nn.silu,
            nn.Conv(self.out_channels, [3, 3], padding="SAME")
        ])
        return final_conv(up12)
