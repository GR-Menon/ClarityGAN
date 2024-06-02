import torch
import torch.nn as nn


class PatchGANBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, has_norm: bool = True):
        super(PatchGANBlock, self).__init__()

        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=1,
                      bias=True)
        ]

        if has_norm:
            layers.append(nn.InstanceNorm2d(num_features=out_channels))

        layers.append(nn.LeakyReLU(negative_slope=0.2))

        self.patch_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.patch_block(x)


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
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=4, stride=1, padding=1,
                                padding_mode="reflect"))

        self.cycle_disc = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.cycle_disc(x))


def test_disc():
    x = torch.randn((16, 3, 256, 256))
    model = CycleDiscriminator()
    preds = model(x)
    print(model)
    print(preds.shape)


if __name__ == '__main__':
    test_disc()
