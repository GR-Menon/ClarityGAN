import os
from typing import Union

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Config:
    data_dir: str
    model_name: str
    model_version: str = 'v1'
    model_dir: str = "./models/"
    log_dir: str = "./logs/"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    img_shape: tuple = (3, 224, 224)

    num_epochs: int = 10
    batch_size: int = 32
    lr: float = 2e-4

    mean: tuple[float, ...] = None
    std: tuple[float, ...] = None
    transforms: "A.Compose" = None

    writer: Union["SummaryWriter", bool] = False

    @property
    def checkpoint_path(self) -> str:
        return f"{self.model_dir}/{self.model_name}/{self.model_version}/"

    @property
    def log_path(self) -> str:
        return f"{self.log_dir}/{self.model_name}/{self.model_version}/"

    def __post_init__(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        if self.writer:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(self.log_path)

        if not self.mean: self.mean = (0.5,) * self.img_shape[0]
        if not self.std: self.std = (0.5,) * self.img_shape[0]
        if not self.transforms: self.transforms = A.Compose([
            A.Resize(self.img_shape[1:]),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255),
            ToTensorV2(),
        ])
