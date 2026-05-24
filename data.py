import os
import zipfile

import pytorch_lightning as pl
import requests
from torch.utils.data import ConcatDataset, DataLoader
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
        self.eval_dataset = getattr(args, "eval_dataset", "synthetic")
        self.eval_data_dir = getattr(args, "eval_data_dir", None)
        self.eval_drop_last = bool(getattr(args, "eval_drop_last", 0))
        self.use_synthetic_train = bool(getattr(args, "use_synthetic_train", 1))
        self.synthetic_train_data_dir = getattr(args, "synthetic_train_data_dir", None)

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
        if self.use_synthetic_train:
            if self.synthetic_train_data_dir is None:
                raise ValueError("--synthetic_train_data_dir must be set when --use_synthetic_train 1")
            synthetic_dataset = self.synthetic_imagefolder_dataset(self.synthetic_train_data_dir, transform)
            dataset = ConcatDataset([dataset, synthetic_dataset])
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def synthetic_imagefolder_dataset(self, root, transform):
        dataset_root = os.path.expanduser(root)
        images_root = os.path.join(dataset_root, "images")
        if os.path.isdir(images_root):
            dataset_root = images_root

        dataset = ImageFolder(root=dataset_root, transform=transform)
        expected_class_to_idx = {f"{idx:03d}": idx for idx in range(43)}
        if dataset.class_to_idx != expected_class_to_idx:
            raise ValueError(
                "Synthetic dataset class directories must be zero-padded 000..042 "
                f"so labels match GTSRB. Got: {dataset.class_to_idx}"
            )
        return dataset

    def eval_transform(self):
        return T.Compose(
            [
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                # T.Normalize(self.mean, self.std),
            ]
        )

    def synthetic_eval_dataset(self, transform):
        if self.eval_data_dir is None:
            raise ValueError("--eval_data_dir must be set when --eval_dataset synthetic")
        return self.synthetic_imagefolder_dataset(self.eval_data_dir, transform)

    def val_dataloader(self):
        transform = self.eval_transform()
        if self.eval_dataset == "synthetic":
            dataset = self.synthetic_eval_dataset(transform)
        elif self.eval_dataset == "gtsrb":
            dataset = GTSRB(root=self.hparams.data_dir, split="test", transform=transform, download=True)
        else:
            raise ValueError(f"Unknown eval dataset: {self.eval_dataset}")

        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=self.eval_drop_last,
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
