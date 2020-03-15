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
from albumentations.augmentations.transforms import ToFloat,CoarseDropout, ElasticTransform

from week7.modular import cfg

sys.path.append('./')
# args = cfg.parser.parse_args(args=[])
global args
args = cfg.args

file_path = args.data


# IPYNB_ENV = True  # By default ipynb notebook env


def preprocess_data_albumentations(mean_tuple, std_tuple):
    """
    Used for pre-processing the data
    when for args.use_albumentations True
    """
    # Train Phase transformations
    global args
    train_transforms = Compose([
        ToFloat(),
        RandomCrop(height=32, width=32, always_apply=True),
        CoarseDropout(max_holes=8,
                      max_height=16,
                      max_width=16,
                      min_holes=0,
                      min_height=0,
                      min_width=0,
                      fill_value=mean_tuple, always_apply=True),
        ElasticTransform(),
        HorizontalFlip(p=0.5),
        ToTensorV2(),
        Normalize(mean=mean_tuple, std=std_tuple),
    ])

    # Test Phase transformations
    test_transforms = Compose([
        ToFloat(),
        ToTensorV2(),
        Normalize(mean=mean_tuple, std=std_tuple),
    ])
    if args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(file_path, train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR10(file_path, train=False, download=True, transform=test_transforms)
    elif args.dataset == 'MNIST':
        train_dataset = datasets.MNIST(file_path, train=True, download=True, transform=train_transforms)
        test_dataset = datasets.MNIST(file_path, train=False, download=True, transform=test_transforms)

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