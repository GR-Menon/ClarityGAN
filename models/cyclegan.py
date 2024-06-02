import torch
import torch.nn as nn

from layer_blocks import PatchGANBlock, ConvBlock, ResidualBlock


class CycleDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, features=None):
        super(CycleDiscriminator, self).__init__()

        if features is None:
            features = [64, 128, 256, 512]

        layers = []
        for feature in features:
            layers.append(
                PatchGANBlock(in_channels=in_channels, out_channels=feature, stride=1 if feature == features[-1] else 2,
                              has_norm=False if feature == features[0] else True))

            in_channels = feature
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))

        self.cycle_disc = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.cycle_disc(x))


class CycleGenerator(nn.Module):
    def __init__(self, img_channels: int = 3, latent_dim: int = 64, num_residuals: int = 9):
        super(CycleGenerator, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=latent_dim, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
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

        self.head = nn.Conv2d(in_channels=latent_dim, out_channels=img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.base(x)

        for layer in self.down_blocks:
            x = layer(x)

        x = self.residual_blocks(x)

        for layer in self.up_blocks:
            x = layer(x)

        x = self.head(x)

        return torch.tanh(x)


def test_disc():
    x = torch.randn((16, 3, 256, 256))
    model = CycleDiscriminator()
    preds = model(x)
    print(model)
    print(preds.shape)


def test_gen():
    x = torch.randn((16, 3, 256, 256))
    model = CycleGenerator(3, 9)
    preds = model(x)
    print(model)
    print(preds.shape)


if __name__ == '__main__':
    choice = int(input("0 for DiscTest, 1 for GenTest: "))
    if choice == 0:
        test_disc()
    elif choice == 1:
        test_gen()
    else:
        pass
