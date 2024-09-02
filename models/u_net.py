#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:52:58 2024

@author: avijitmajhi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Conv, Down_Block, Up_Block

class UNet(nn.Module):

    def __init__(self, device,input_length, output_length, filter_number=64):
        super(UNet, self).__init__()
        
        self.device = device
        

        self.conv_1_1 = Conv(input_length, filter_number, bn=True)
        self.conv_1_2 = Conv(filter_number, filter_number, bn=True)
        self.down_1 = Down_Block(filter_number, filter_number*2, bn=True)
        self.down_2 = Down_Block(filter_number*2, filter_number*4, bn=True)
        self.down_3 = Down_Block(filter_number*4, filter_number*8, bn=True)
        self.down_4 = Down_Block(filter_number*8, filter_number*16, bn=True)

        self.up_1 = Up_Block(16*filter_number, 8*filter_number, bn=True)
        self.up_2 = Up_Block(8*filter_number, 4*filter_number, bn=True)
        self.up_3 = Up_Block(4*filter_number, 2*filter_number, bn=True)
        self.up_4 = Up_Block(2*filter_number, filter_number, bn=True)

        self.out = nn.Conv2d(filter_number, output_length, kernel_size=1)

    def forward(self, x):

        x = self.conv_1_1(x)
        x_1 = self.conv_1_2(x)
        x_2 = self.down_1(x_1)
        x_3 = self.down_2(x_2)
        x_4 = self.down_3(x_3)
        x_5 = self.down_4(x_4)

        x = self.up_1(x_5, x_4)
        x = self.up_2(x, x_3)
        x = self.up_3(x, x_2)
        x = self.up_4(x, x_1)

        x = self.out(x)

        return x
