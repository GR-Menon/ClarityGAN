import torch.nn as nn


class PatchGANBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, has_norm: bool = True):
        super(PatchGANBlock, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False)
        ]

        if has_norm:
            layers.append(nn.InstanceNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.patch_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.patch_block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True, use_act: bool = True, use_dropout:bool=False, **kwargs):
        super(ConvBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if downsample
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )
        if use_dropout:
            self.conv_block = nn.Sequential(self.conv_block, nn.Dropout(0.5))

    def forward(self, x):
        return self.conv_block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels:int):
        super(ResidualBlock, self).__init__()
        self.residual_block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, kernel_size=3, padding=1, use_act=False),
        )

    def forward(self, x):
        return x + self.residual_block(x)
