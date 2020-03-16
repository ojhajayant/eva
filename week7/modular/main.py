#!/usr/bin/env python
"""
main.py: This is the main script to be run to either train or make inference.
example usage is as below:
python main.py train --SEED 2 --batch_size 64  --epochs 10 --lr 0.01 \
                     --dropout 0.05 --l1_weight 0.00002  --l2_weight_decay 0.000125 \
                     --L1 True --L2 False --data data --best_model_path saved_models \
                     --prefix data
python main.py test  --batch_size 64  --data data --best_model_path saved_models \
                     --best_model 'CIFAR10_model_epoch-39_L1-1_L2-0_val_acc-81.83.h5' \
                     --prefix data
"""
from __future__ import print_function

import os
import sys
import warnings

import numpy as np
import torch
import torch.optim as optim
from torchsummary import summary

from week7.modular import cfg
from week7.modular.models import resnet18, s5_s6_custom_model_mnist, s7_custom_model_cifar10, QuizDNN
from week7.modular import preprocess
from week7.modular import preprocess_albumentations
from week7.modular import test
from week7.modular import train
from week7.modular import utils

sys.path.append('./')
# args = cfg.parser.parse_args(args=[])
global args
args = cfg.args
if args.cmd == None:
    args.cmd = 'train'

import sys

import torch.nn as nn
import torch.nn.functional as F

from week7.modular import cfg

sys.path.append('./')


def main_s8_resnet():
    global args
    print("The config used for this run are being saved @ {}".format(os.path.join(args.prefix, 'config_params.txt')))
    utils.write(vars(args), os.path.join(args.prefix, 'config_params.txt'))
    mean, std = preprocess.get_dataset_mean_std()
    if args.use_albumentations:
        print("Using albumentation lib for image-augmentation & other transforms")
        train_dataset, test_dataset, train_loader, test_loader = \
            preprocess_albumentations.preprocess_data_albumentations((mean[0], mean[1], mean[2]),
                                                                     (std[0], std[1], std[2]))
    else:
        train_dataset, test_dataset, train_loader, test_loader = \
            preprocess.preprocess_data((mean[0], mean[1], mean[2]), (std[0], std[1], std[2]))

    preprocess.get_data_stats(train_dataset, test_dataset, train_loader)
    utils.plot_train_samples(train_loader)
    L1 = args.L1
    L2 = args.L2
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)
    model = resnet18.ResNet18().to(device)
    if args.dataset == 'CIFAR10':
        summary(model, input_size=(3, 32, 32))
    elif args.dataset == 'MNIST':
        summary(model, input_size=(1, 28, 28))
    if args.cmd == 'train':
        print("Model training starts on {} dataset".format(args.dataset))
        # Enable L2-regularization with supplied value of weight decay, or keep it default-0
        if L2:
            weight_decay = args.l2_weight_decay
        else:
            weight_decay = 0
        # lr = args.lr
        lr = 0.0006
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        EPOCHS = args.epochs
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch + 1)
            train.train(model, device, train_loader, optimizer, epoch)
            test.test(model, device, test_loader, optimizer, epoch)
        utils.plot_acc_loss()
    elif args.cmd == 'test':
        print("Model inference starts on {}  dataset".format(args.dataset))
        # model_name = args.best_model
        model_name = 'CIFAR10_model_epoch-8_L1-1_L2-0_val_acc-81.91.h5'
        print("Loaded the best model: {} from last training session".format(model_name))
        # model = utils.load_model(network.Net(), device, model_name=model_name)#Custom Model used in S7
        model = utils.load_model(resnet18.ResNet18(), device, model_name=model_name)
        y_test = np.array(test_dataset.targets)
        print("The confusion-matrix and classification-report for this model are:")
        y_pred = utils.model_pred(model, device, y_test, test_dataset)
        x_test = test_dataset.data
        utils.display_mislabelled(model, device, x_test, y_test.reshape(-1, 1), y_pred, test_dataset,
                                  title_str='Predicted Vs Actual With L1')


def main_s7_custom_model():
    global args
    print("The config used for this run are being saved @ {}".format(os.path.join(args.prefix, 'config_params.txt')))
    utils.write(vars(args), os.path.join(args.prefix, 'config_params.txt'))
    mean, std = preprocess.get_dataset_mean_std()
    if args.use_albumentations:
        train_dataset, test_dataset, train_loader, test_loader = \
            preprocess_albumentations.preprocess_data_albumentations((mean[0], mean[1], mean[2]),
                                                                     (std[0], std[1], std[2]))
    else:
        train_dataset, test_dataset, train_loader, test_loader = \
            preprocess.preprocess_data((mean[0], mean[1], mean[2]), (std[0], std[1], std[2]))
    preprocess.get_data_stats(train_dataset, test_dataset, train_loader)
    utils.plot_train_samples(train_loader)
    L1 = args.L1
    L2 = args.L2
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)
    model = s7_custom_model_cifar10.Net().to(device)
    if args.dataset == 'CIFAR10':
        summary(model, input_size=(3, 32, 32))
    elif args.dataset == 'MNIST':
        summary(model, input_size=(1, 28, 28))
    if args.cmd == 'train':
        print("Model training starts on {} dataset".format(args.dataset))
        # Enable L2-regularization with supplied value of weight decay, or keep it default-0
        if L2:
            weight_decay = args.l2_weight_decay
        else:
            weight_decay = 0
        # lr = args.lr
        lr = 0.01
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        EPOCHS = args.epochs
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch + 1)
            train.train(model, device, train_loader, optimizer, epoch)
            test.test(model, device, test_loader, optimizer, epoch)
        utils.plot_acc_loss()
    elif args.cmd == 'test':
        print("Model inference starts on {}  dataset".format(args.dataset))
        # model_name = args.best_model
        model_name = 'CIFAR10_model_epoch-2_L1-1_L2-0_val_acc-62.28.h5'
        print("Loaded the best model: {} from last training session".format(model_name))
        model = utils.load_model(s7_custom_model_cifar10.Net(), device, model_name=model_name)
        y_test = np.array(test_dataset.targets)
        print("The confusion-matrix and classification-report for this model are:")
        y_pred = utils.model_pred(model, device, y_test, test_dataset)
        x_test = test_dataset.data
        utils.display_mislabelled(model, device, x_test, y_test.reshape(-1, 1), y_pred, test_dataset,
                                  title_str='Predicted Vs Actual With L1')


def main_s6_custom_model():
    global args
    print("The config used for this run are being saved @ {}".format(os.path.join(args.prefix, 'config_params.txt')))
    utils.write(vars(args), os.path.join(args.prefix, 'config_params.txt'))

    mean, std = preprocess.get_dataset_mean_std()
    if not isinstance(mean, tuple):
        train_dataset, test_dataset, train_loader, test_loader = preprocess.preprocess_data((mean,), (std,))
    else:
        if args.use_albumentations:
            train_dataset, test_dataset, train_loader, test_loader = \
                preprocess_albumentations.preprocess_data_albumentations((mean[0], mean[1], mean[2]),
                                                                         (std[0], std[1], std[2]))
        else:
            train_dataset, test_dataset, train_loader, test_loader = \
                preprocess.preprocess_data((mean[0], mean[1], mean[2]), (std[0], std[1], std[2]))
    preprocess.get_data_stats(train_dataset, test_dataset, train_loader)
    utils.plot_train_samples(train_loader)
    L1 = args.L1
    L2 = args.L2
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)
    model = s5_s6_custom_model_mnist.Net().to(device)
    if args.dataset == 'CIFAR10':
        summary(model, input_size=(3, 32, 32))
    elif args.dataset == 'MNIST':
        summary(model, input_size=(1, 28, 28))
    if args.cmd == 'train':
        print("Model training starts on {} dataset".format(args.dataset))
        # Enable L2-regularization with supplied value of weight decay, or keep it default-0
        if L2:
            weight_decay = args.l2_weight_decay
        else:
            weight_decay = 0
        # lr = args.lr
        lr = 0.01
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        EPOCHS = args.epochs
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch + 1)
            train.train(model, device, train_loader, optimizer, epoch)
            test.test(model, device, test_loader, optimizer, epoch)
        utils.plot_acc_loss()
    elif args.cmd == 'test':
        print("Model inference starts on {}  dataset".format(args.dataset))
        # model_name = args.best_model
        model_name = 'MNIST_model_epoch-8_L1-1_L2-0_val_acc-99.26.h5'
        print("Loaded the best model: {} from last training session".format(model_name))
        model = utils.load_model(s5_s6_custom_model_mnist.Net(), device, model_name=model_name)
        y_test = np.array(test_dataset.targets)
        print("The confusion-matrix and classification-report for this model are:")
        y_pred = utils.model_pred(model, device, y_test, test_dataset)
        x_test = test_dataset.data
        utils.display_mislabelled(model, device, x_test, y_test.reshape(-1, 1), y_pred, test_dataset,
                                  title_str='Predicted Vs Actual With L1')


def main_QuizDNN_model():
    global args
    print("The config used for this run are being saved @ {}".format(os.path.join(args.prefix, 'config_params.txt')))
    utils.write(vars(args), os.path.join(args.prefix, 'config_params.txt'))
    mean, std = preprocess.get_dataset_mean_std()
    if args.use_albumentations:
        train_dataset, test_dataset, train_loader, test_loader = \
            preprocess_albumentations.preprocess_data_albumentations((mean[0], mean[1], mean[2]),
                                                                     (std[0], std[1], std[2]))
    else:
        train_dataset, test_dataset, train_loader, test_loader = \
            preprocess.preprocess_data((mean[0], mean[1], mean[2]), (std[0], std[1], std[2]))
    preprocess.get_data_stats(train_dataset, test_dataset, train_loader)
    utils.plot_train_samples(train_loader)
    L1 = args.L1
    L2 = args.L2
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)
    model = QuizDNN.QuizDNN().to(device)
    if args.dataset == 'CIFAR10':
        summary(model, input_size=(3, 32, 32))
    elif args.dataset == 'MNIST':
        summary(model, input_size=(1, 28, 28))
    if args.cmd == 'train':
        print("Model training starts on {} dataset".format(args.dataset))
        # Enable L2-regularization with supplied value of weight decay, or keep it default-0
        if L2:
            weight_decay = args.l2_weight_decay
        else:
            weight_decay = 0
        # lr = args.lr
        lr = 0.01
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        EPOCHS = args.epochs
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch + 1)
            train.train(model, device, train_loader, optimizer, epoch)
            test.test(model, device, test_loader, optimizer, epoch)
        utils.plot_acc_loss()
    elif args.cmd == 'test':
        print("Model inference starts on {}  dataset".format(args.dataset))
        # model_name = args.best_model
        model_name = 'CIFAR10_model_epoch-2_L1-1_L2-0_val_acc-62.28.h5'
        print("Loaded the best model: {} from last training session".format(model_name))
        model = utils.load_model(QuizDNN.QuizDNN(), device, model_name=model_name)
        y_test = np.array(test_dataset.targets)
        print("The confusion-matrix and classification-report for this model are:")
        y_pred = utils.model_pred(model, device, y_test, test_dataset)
        x_test = test_dataset.data
        utils.display_mislabelled(model, device, x_test, y_test.reshape(-1, 1), y_pred, test_dataset,
                                  title_str='Predicted Vs Actual With L1')


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    # --------
    # args.dataset = 'MNIST'
    # args.cmd ='test'
    # args.IPYNB_ENV = 'False'
    # main_s6_custom_model()
    # --------
    # args.dataset = 'CIFAR10'
    # args.cmd = 'test'
    # args.IPYNB_ENV = 'False'
    # args.epochs = 2
    # main_s7_custom_model()
    # --------
    args.dataset = 'CIFAR10'
    args.cmd = 'train'
    args.IPYNB_ENV = 'False'
    args.epochs = 40
    args.use_albumentations = True
    main_s8_resnet()
    # --------END
