import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader


class ResideDataset(Dataset):
    def __init__(self, root: str, img_transform: T.Compose = None):
        super(ResideDataset, self).__init__()

        self.root = root
        self.haze_dir = os.path.join(root, 'hazy')
        self.clear_dir = os.path.join(root, 'clear')
        self.haze_imgs = [os.path.join(root, 'hazy', img) for img in os.listdir(self.haze_dir)]
        self.img_transform = img_transform

    def __len__(self):
        return len(self.haze_imgs)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        haze = Image.open(self.haze_imgs[idx])

        clear_id = self.haze_imgs[idx].split('/')[-1].split('_')[0]
        clear = Image.open(os.path.join(self.clear_dir, clear_id + ".png"))

        haze, clear = haze.convert('RGB'), clear.convert('RGB')

        if self.img_transform:
            haze, clear = self.img_transform(haze), self.img_transform(clear)
        return haze, clear


def test_reside():
    train_path = "../RESIDE-Std"
    test_path = "../RESIDE-Synth/indoor"

    train_transforms = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    test_transforms = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    train_dataset = ResideDataset(train_path, img_transform=train_transforms)
    test_dataset = ResideDataset(test_path, img_transform=test_transforms)

    print(len(train_dataset), len(test_dataset))
    train_img, train_target = train_dataset[0]
    test_img, test_target = test_dataset[0]

    plots = [train_img, train_target, test_img, test_target]
    plt.figure(figsize=(8, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.axis(False)
        plt.title(plots[i].shape)
        plt.imshow(plots[i].permute(1, 2, 0))
    plt.show()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    img1, img2 = next(iter(train_loader))
    print(img1.shape, img2.shape)


if __name__ == '__main__':
    test_reside()