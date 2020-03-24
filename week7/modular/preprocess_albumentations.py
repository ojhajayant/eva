#!/usr/bin/env python
"""
preprocess_albumentations.py: This contains the data-pre-processing routine
implemented using albumentations library.
https://github.com/albumentations-team/albumentations
"""
from __future__ import print_function

import sys

import numpy as np
import torch
from albumentations import Compose, RandomCrop, HorizontalFlip
from albumentations.augmentations.transforms import ToFloat, CoarseDropout, PadIfNeeded, GaussianBlur, VerticalFlip, Rotate
from albumentations.pytorch import ToTensor
from torchvision import datasets

from week7.modular import cfg

sys.path.append('./')
global args
args = cfg.args

file_path = args.data


# IPYNB_ENV = True  # By default ipynb notebook env
# Thanks to Harsha V (EVA4 group) for the piece below!
class album_Compose:
    def __init__(self,
                 train=True,
                 mean=[0.49139968, 0.48215841, 0.44653091],
                 std=[0.24703223, 0.24348513, 0.26158784]
                 ):
        if train:
            self.albumentattions_transform = Compose([
                PadIfNeeded(min_height=32, min_width=32, border_mode=0, value=[0, 0, 0], always_apply=True),
                # Cutout(num_holes=3, max_h_size=4, max_w_size=4, p=0.5),
                CoarseDropout(max_holes=8, max_height=4, max_width=4, p=0.5, fill_value=tuple([x * 255.0 for x in mean])),
                # ElasticTransform()
                HorizontalFlip(p=0.5),
                # VerticalFlip(p=0.5),
                Rotate(limit=10),
                # GaussianBlur(p=0.5),
                RandomCrop(height=32, width=32, always_apply=True),
                ToFloat(max_value=None, always_apply=True),
                ToTensor(normalize={'mean': list(mean), 'std': list(std)}),
                # ToTensorV2(normalize={'mean': list(mean), 'std': list(std)}),
            ])
        else:
            self.albumentattions_transform = Compose([
                ToFloat(max_value=None, always_apply=True),
                ToTensor(normalize={'mean': list(mean), 'std': list(std)}),
                # ToTensorV2(normalize={'mean': list(mean), 'std': list(std)}),
            ])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentattions_transform(image=img)['image']
        return img


def preprocess_data_albumentations(mean_tuple, std_tuple):
    """
    Used for pre-processing the data
    when for args.use_albumentations True
    """
    # Train Phase transformations
    global args
    # tensor_args1 = dict(always_apply=True, p=1.0)
    tensor_args2 = dict(num_classes=1, sigmoid=True, normalize=None)
    norm_args = dict(mean=mean_tuple, std=std_tuple, max_pixel_value=255.0, always_apply=False, p=1.0)
    print("************")
    train_transforms = album_Compose(train=True, mean=mean_tuple, std=std_tuple)

    # Test Phase transformations
    test_transforms = album_Compose(train=False, mean=mean_tuple, std=std_tuple)

    train_kwargs = dict(train=True, download=True, transform=train_transforms)
    test_kwargs = dict(train=False, download=True, transform=test_transforms)
    if args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(file_path, **train_kwargs)
        test_dataset = datasets.CIFAR10(file_path, **test_kwargs)
    elif args.dataset == 'MNIST':
        train_dataset = datasets.MNIST(file_path, **train_kwargs)
        test_dataset = datasets.MNIST(file_path, **test_kwargs)

    print("CUDA Available?", args.cuda)

    # For reproducibility
    torch.manual_seed(args.SEED)

    if args.cuda:
        torch.cuda.manual_seed(args.SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=args.batch_size, num_workers=4,
                           pin_memory=True) if args.cuda else dict(shuffle=True,
                                                                   batch_size=args.batch_size)
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    # test dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)
    return train_dataset, test_dataset, train_loader, test_loader
