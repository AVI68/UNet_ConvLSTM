#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 22:25:51 2024

@author: avijitmajhi

RainNet Model
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
class RainNet(nn.Module):
    def __init__(self, input_channels=4, mode="regression"):
        super(RainNet, self).__init__()
        
        # Define the encoder part of the network
        self.conv1f = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv1s = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2f = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2s = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3f = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3s = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4f = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4s = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(0.5)
        
        self.conv5f = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv5s = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.drop5 = nn.Dropout(0.5)
        
        # Define the decoder part of the network
        self.up6 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        
        self.up7 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        
        self.up8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        
        self.up9 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        self.final_conv = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        
        if mode == "regression":
            self.output_conv = nn.Conv2d(2, 1, kernel_size=1)
        elif mode == "segmentation":
            self.output_conv = nn.Conv2d(2, 1, kernel_size=1)
            self.sigmoid = nn.Sigmoid()
        
        self.mode = mode
    
    def forward(self, x):
        # Encoder
        x1 = F.relu(self.conv1f(x))
        x1 = F.relu(self.conv1s(x1))
        x2 = self.pool1(x1)
        
        x2 = F.relu(self.conv2f(x2))
        x2 = F.relu(self.conv2s(x2))
        x3 = self.pool2(x2)
        
        x3 = F.relu(self.conv3f(x3))
        x3 = F.relu(self.conv3s(x3))
        x4 = self.pool3(x3)
        
        x4 = F.relu(self.conv4f(x4))
        x4 = F.relu(self.conv4s(x4))
        x4 = self.drop4(x4)
        x5 = self.pool4(x4)
        
        x5 = F.relu(self.conv5f(x5))
        x5 = F.relu(self.conv5s(x5))
        x5 = self.drop5(x5)
        
        # Decoder
        x6 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True)
        x6 = torch.cat([x6, x4], dim=1)
        x6 = F.relu(self.conv6(F.relu(self.up6(x6))))
        
        x7 = F.interpolate(x6, scale_factor=2, mode='bilinear', align_corners=True)
        x7 = torch.cat([x7, x3], dim=1)
        x7 = F.relu(self.conv7(F.relu(self.up7(x7))))
        
        x8 = F.interpolate(x7, scale_factor=2, mode='bilinear', align_corners=True)
        x8 = torch.cat([x8, x2], dim=1)
        x8 = F.relu(self.conv8(F.relu(self.up8(x8))))
        
        x9 = F.interpolate(x8, scale_factor=2, mode='bilinear', align_corners=True)
        x9 = torch.cat([x9, x1], dim=1)
        x9 = F.relu(self.conv9(F.relu(self.up9(x9))))
        
        x9 = F.relu(self.final_conv(x9))
        
        if self.mode == "regression":
            return self.output_conv(x9)
        elif self.mode == "segmentation":
            return self.sigmoid(self.output_conv(x9))

# Loss function equivalent to log_cosh
def log_cosh_loss(y_pred, y_true):
    loss = torch.log(torch.cosh(y_pred - y_true))
    return torch.mean(loss)