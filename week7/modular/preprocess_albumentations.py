#!/usr/bin/env python
"""
preprocess_albumentations.py: This contains the data-pre-processing routine
implemented using albumentations library.
https://github.com/albumentations-team/albumentations
"""
from __future__ import print_function

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets
from torchvision.datasets import CIFAR10, MNIST
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import ToFloat, CoarseDropout, ElasticTransform
from albumentations.pytorch import ToTensor

from week7.modular import cfg

sys.path.append('./')
# args = cfg.parser.parse_args(args=[])
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
                RandomCrop(height=32, width=32, always_apply=True),
                # ElasticTransform(),
                HorizontalFlip(p=0.5),
                Normalize(mean=mean,
                          std=std),
                CoarseDropout(max_holes=8,
                              max_height=8,
                              max_width=8,
                              min_holes=1,
                              min_height=1,
                              min_width=1,
                              fill_value=mean, always_apply=False),
                ToTensorV2()
            ])
        else:
            self.albumentattions_transform = Compose([
                Normalize(mean=mean,
                          std=std),
                ToTensorV2()
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
