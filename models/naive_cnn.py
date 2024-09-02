#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:54:55 2024

@author: avijitmajhi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Conv,ConvTranspose

class cnn_2D(nn.Module):

    def __init__(self, device, input_length, output_length, filter_number):
        super(cnn_2D, self).__init__()
        self.device = device

        self.enc1 = Conv(in_channels=input_length, out_channels=filter_number, kernel_size=(4, 4), stride=2, padding=1, bn=True)
        self.enc2 = Conv(in_channels=filter_number, out_channels=filter_number, kernel_size=(4, 4), stride=2, padding=1, bn=True)
        self.enc3 = Conv(in_channels=filter_number, out_channels=8*filter_number, kernel_size=(4, 4), stride=2, padding=1, bn=True)
        self.enc4 = Conv(in_channels=8*filter_number, out_channels=12*filter_number, kernel_size=(4, 4), stride=2, padding=1, bn=True)
        self.enc5 = Conv(in_channels=12*filter_number, out_channels=16*filter_number, kernel_size=(4, 4), stride=2, padding=1, bn=True)

        self.vid1 = ConvTranspose(in_channels=16*filter_number, out_channels=16*filter_number, kernel_size=(1, 1), stride=1, padding=0, bn=True)
        self.vid2 = ConvTranspose(in_channels=16*filter_number, out_channels=16*filter_number, kernel_size=(4, 4), stride=2, padding=1, bn=True)
        self.vid3 = ConvTranspose(in_channels=16*filter_number, out_channels=8*filter_number, kernel_size=(4, 4), stride=2, padding=1, bn=True)
        self.vid4 = ConvTranspose(in_channels=8*filter_number, out_channels=4*filter_number, kernel_size=(4, 4), stride=2, padding=1, bn=True)
        self.vid5 = ConvTranspose(in_channels=4*filter_number, out_channels=24, kernel_size=(4, 4), stride=2, padding=1, bn=True)
        self.vid6 = nn.ConvTranspose2d(in_channels=24, out_channels=output_length, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, x):

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        x = self.vid1(x)
        x = self.vid2(x)
        x = self.vid3(x)
        x = self.vid4(x)
        x = self.vid5(x)
        x = F.relu(self.vid6(x))

        return x
