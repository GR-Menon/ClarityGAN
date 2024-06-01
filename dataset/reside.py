import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ResideDataset(Dataset):
    def __init__(self, root, img_size, train: bool = True, format='.png', transform: transforms.Compose = None):
        super(ResideDataset, self).__init__()

        self.img_size = img_size
        self.train = train
        self.format = format
        self.transform = transform
        self.haze_imgs = [os.path.join(root, 'hazy', img) for img in os.listdir(os.path.join(root, 'hazy'))]
        self.clear_dir = os.path.join(root, 'clear')

    def __len__(self):
        return len(self.haze_imgs)

    def __getitem__(self, index):
        haze = np.array(Image.open(self.haze_imgs[index]))
        #
        # if isinstance(self.img_size, int):
        #     while haze.size[0] < self.img_size or haze.size[1] < self.img_size:
        #         index = random.randint(0, 20000)
        #         haze = Image.open(self.haze_imgs[index])

        img = self.haze_imgs[index]
        id = img.split('/')[-1].split('_')[0]

        clear_name = id + self.format
        clear = np.array(Image.open(os.path.join(self.clear_dir, clear_name)))
        # clear = transforms.CenterCrop(haze.size[::-1])(clear)
        #
        # if not isinstance(self.img_size, str):
        #     i, j, h, w = transforms.RandomCrop.get_params(haze, output_size=(self.img_size, self.img_size))
        #     haze = TF.crop(haze, i, j, h, w)
        #     clear = TF.crop(clear, i, j, h, w)
        if self.transform:
            haze, clear = self.transform(haze.convert('RGB'), clear.convert('RGB'))
        return haze, clear

    # def augmentations(self, data, target):
    #     if self.train:
    #         rand_hor = random.randint(0, 1)
    #         rand_rot = random.randint(0, 3)
    #         data = transforms.RandomHorizontalFlip(rand_hor)(data)
    #         target = transforms.RandomHorizontalFlip(rand_hor)(target)
    #         if rand_rot:
    #             data = TF.rotate(data, 90 * rand_rot)
    #             target = TF.rotate(target, 90 * rand_rot)
    #     data = transforms.ToTensor()(data)
    #     data = transforms.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
    #     target = transforms.ToTensor()(target)
    #     return data, target


def test_reside():
    train_path = "../RESIDE-Std"
    test_path = "../RESIDE-Synth/indoor"

    train_dataset = ResideDataset(train_path, train=True, img_size=240)
    test_dataset = ResideDataset(test_path, train=False, img_size='whole_img')

    print(len(train_dataset), len(test_dataset))
    img, target = train_dataset[0]
    print(img.shape, target.shape)


if __name__ == '__main__':
    test_reside()
