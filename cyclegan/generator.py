import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True, use_act: bool = True,
                 use_dropout: bool = False, **kwargs):
        super(ConvBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding_mode="reflect", **kwargs)

            if downsample
            else nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, **kwargs),

            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )
        if use_dropout:
            self.conv_block = nn.Sequential(self.conv_block, nn.Dropout(p=0.5))

    def forward(self, x):
        return self.conv_block(x)


class ResidualBlock(nn.Module):
    def __init__(self, features: int):
        super(ResidualBlock, self).__init__()
        self.residual_block = nn.Sequential(
            ConvBlock(in_channels=features, out_channels=features, kernel_size=3, padding=1),
            ConvBlock(in_channels=features, out_channels=features, kernel_size=3, padding=1, use_act=False),
        )

    def forward(self, x):
        return x + self.residual_block(x)


class CycleGenerator(nn.Module):
    def __init__(self, img_channels: int = 3, latent_dim: int = 64, num_residuals: int = 9):
        super(CycleGenerator, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=latent_dim, kernel_size=7, stride=1, padding=3,
                      padding_mode="reflect"),
            nn.ReLU(inplace=True)
        )

        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(in_channels=latent_dim, out_channels=latent_dim * 2, kernel_size=3, stride=2, padding=1),
                ConvBlock(in_channels=latent_dim * 2, out_channels=latent_dim * 4, kernel_size=3, stride=2, padding=1),
            ]
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(features=latent_dim * 4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(in_channels=latent_dim * 4, out_channels=latent_dim * 2, kernel_size=3, stride=2, padding=1,
                          output_padding=1,
                          downsample=False),
                ConvBlock(in_channels=latent_dim * 2, out_channels=latent_dim, kernel_size=3, stride=2, padding=1,
                          output_padding=1,
                          downsample=False),
            ]
        )

        self.head = nn.Conv2d(in_channels=latent_dim, out_channels=img_channels, kernel_size=7, stride=1, padding=3,
                              padding_mode="reflect")

    def forward(self, x):
        x = self.base(x)

        for layer in self.down_blocks:
            x = layer(x)

        x = self.residual_blocks(x)

        for layer in self.up_blocks:
            x = layer(x)

        x = self.head(x)

        return torch.tanh(x)


def test_gen():
    x = torch.randn((16, 3, 256, 256))
    model = CycleGenerator(3, 9)
    preds = model(x)
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test_gen()
