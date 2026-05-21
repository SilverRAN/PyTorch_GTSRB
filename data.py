import os
import zipfile

import pytorch_lightning as pl
import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, GTSRB, ImageFolder
from tqdm import tqdm


class GTSRBData(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        # self.mean = (0.485, 0.456, 0.406)  # ImageNet mean
        # self.std = (0.229, 0.224, 0.225)   # ImageNet std

        # If you set image size by yourself
        self.img_size = args.img_size if hasattr(args, 'img_size') else 48

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.Resize((self.img_size, self.img_size)),  # 调整尺寸
                T.RandomCrop(self.img_size, padding=4),
                # T.RandomHorizontalFlip(), # 交通标志不做水平翻转!
                T.ToTensor(),
                # T.Normalize(self.mean, self.std),
            ]
        )
        dataset = GTSRB(root=self.hparams.data_dir, split="train", transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                # T.Normalize(self.mean, self.std),
            ]
        )
        dataset = GTSRB(root=self.hparams.data_dir, split="test", transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

# class GTSRBData(pl.LightningDataModule): # Load dataset from ImageFolder
#     def __init__(self, args):
#         super().__init__()
#         self.save_hyperparameters(args)
#         # 根据你的数据集计算均值和标准差，或使用 ImageNet 的标准值
#         self.mean = (0.485, 0.456, 0.406)  # ImageNet mean
#         self.std = (0.229, 0.224, 0.225)   # ImageNet std
        
#         # 如果需要自定义图像尺寸
#         self.img_size = args.img_size if hasattr(args, 'img_size') else 48

#     def train_dataloader(self):
#         transform = T.Compose([
#             T.Resize((self.img_size, self.img_size)),  # 调整尺寸
#             T.RandomCrop(self.img_size, padding=4),
#             # T.RandomHorizontalFlip(), # 交通标志不做水平翻转!
#             T.ToTensor(),
#             # T.Normalize(self.mean, self.std),
#         ])
#         dataset = ImageFolder(
#             root=os.path.join(self.hparams.data_dir, 'train'),
#             transform=transform
#         )
#         dataloader = DataLoader(
#             dataset,
#             batch_size=self.hparams.batch_size,
#             num_workers=self.hparams.num_workers,
#             shuffle=True,
#             drop_last=True,
#             pin_memory=True,
#         )
#         return dataloader

#     def val_dataloader(self):
#         transform = T.Compose([
#             T.Resize((self.img_size, self.img_size)),
#             T.ToTensor(),
#             # T.Normalize(self.mean, self.std),
#         ])
#         dataset = ImageFolder(
#             root=os.path.join(self.hparams.data_dir, 'test'),
#             transform=transform
#         )
#         dataloader = DataLoader(
#             dataset,
#             batch_size=self.hparams.batch_size,
#             num_workers=self.hparams.num_workers,
#             drop_last=True,
#             pin_memory=True,
#         )
#         return dataloader

#     def test_dataloader(self):
#         return self.val_dataloader()