#!/usr/bin/env python
"""
QuizDNN.py

x1 = Input
x2 = Conv(x1)
x3 = Conv(x1 + x2)
x4 = MaxPooling(x1 + x2 + x3)
x5 = Conv(x4)
x6 = Conv(x4 + x5)
x7 = Conv(x4 + x5 + x6)
x8 = MaxPooling(x5 + x6 + x7)
x9 = Conv(x8)
x10 = Conv (x8 + x9)
x11 = Conv (x8 + x9 + x10)
x12 = GAP(x11)
x13 = FC(x12)
"""
from __future__ import print_function

import sys

import torch.nn as nn
import torch.nn.functional as F

from week7.modular import cfg

sys.path.append('./')

args = cfg.parser.parse_args(args=[])
dropout_value = args.dropout

dropout_value = 0.05


class QuizDNN(nn.Module):
    def __init__(self):
        super(QuizDNN, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )

        self.pool1 = nn.MaxPool2d(2, 2)
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )

        self.pool2 = nn.MaxPool2d(2, 2)
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:10x10x16, output:8x8x16, RF:14x14

        # TRANSITION BLOCK 3
        self.pool3 = nn.MaxPool2d(2, 2)  # input:24x24x8, output:12x12x8, RF:6x6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:12x12x8, output:12x12x8, RF:6x6

        # C4 Block
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Dropout(dropout_value),
            nn.ReLU(),
        )  # input:8x8x16, output:6x6x16, RF:18x18

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )  # input:6x6x16, output:1x1x16, RF:32x32

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )  # input:1x1x16, output:1x1x10,

    def forward(self, x1):
        x2 = self.convblock1(x1)
        x3 = self.convblock2(x1+ x2)
        x4 = self.pool1(x1 + x2 + x3)
        x5 = self.convblock3(x4)
        x6 = self.convblock4(x4 + x5)
        x7 = self.convblock5(x4 + x5 + x6)
        x8 = self.pool2(x5 + x6 + x7)
        x9 = self.convblock6(x8)
        x10 = self.convblock7(x8 + x9)
        x11 = self.convblock8(x8 + x9 + x10)
        x12 = self.gap(x11)
        x13 = self.convblock9(x12)
        # Reshape
        x13 = x13.view(-1, 10)
        return F.log_softmax(x13, dim=-1)  # torch.nn.CrossEntropyLoss:criterion combines
        # nn.LogSoftmax() and nn.NLLLoss() in one single class.        # nn.LogSoftmax() and nn.NLLLoss() in one single class.