#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:52:58 2024

@author: avijitmajhi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import Conv, TrajGRU_cell, ConvTranspose

class TrajGRU(nn.Module):

    def __init__(self, device, input_length=16, output_length=15):
        super(TrajGRU, self).__init__()

        self.sequence_length = input_length
        self.output_length = output_length
        self.device = device

        channels_input = 1

        # Downsample the spatial dimensions
        self.conv_1 = Conv(in_channels=channels_input, out_channels=8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.traj_gru_1 = TrajGRU_cell(input_size=(8, 128, 128), hidden_filters=64, L=13, sequence_length=input_length, device=self.device)
        self.conv_2 = Conv(in_channels=64, out_channels=192, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.traj_gru_2 = TrajGRU_cell(input_size=(192, 64, 64), hidden_filters=192, L=13, sequence_length=input_length, device=self.device)
        self.conv_3 = Conv(in_channels=192, out_channels=192, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.traj_gru_3 = TrajGRU_cell(input_size=(192, 32, 32), hidden_filters=192, L=9, sequence_length=input_length, device=self.device)

        # Upsample the spatial dimensions
        self.traj_gru_4 = TrajGRU_cell(input_size=(192, 32, 32), hidden_filters=192, L=9, sequence_length=input_length, device=self.device)
        self.conv_transpose_1 = ConvTranspose(in_channels=192, out_channels=192, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.traj_gru_5 = TrajGRU_cell(input_size=(192, 64, 64), hidden_filters=192, L=13, sequence_length=input_length, device=self.device)
        self.conv_transpose_2 = ConvTranspose(in_channels=192, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.traj_gru_6 = TrajGRU_cell(input_size=(64, 128, 128), hidden_filters=64, L=13, sequence_length=input_length, device=self.device)
        self.conv_transpose_3 = ConvTranspose(in_channels=64, out_channels=8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.out_conv = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))


    def forward(self, x):

        # Encoder
        output, state_1 = self.forward_encoder(self.conv_1, self.traj_gru_1, x)
        output, state_2 = self.forward_encoder(self.conv_2, self.traj_gru_2, output)
        output, state_3 = self.forward_encoder(self.conv_3, self.traj_gru_3, output)

        # Forecaster
        output = self.forward_forecaster(self.conv_transpose_1, self.traj_gru_4, None, state_3)
        output = self.forward_forecaster(self.conv_transpose_2, self.traj_gru_5, output, state_2)
        output = self.forward_forecaster(self.conv_transpose_3, self.traj_gru_6, output, state_1)

        output = torch.reshape(output, (-1, output.size(2), output.size(3), output.size(4)))
        output = F.relu(self.out_conv(output))
        output = torch.reshape(output, (output.size(0) // self.sequence_length, self.sequence_length, output.size(1), output.size(2), output.size(3)))

        # Adjust sequence length from 16 to 15 for the output
        output = output[:, :self.output_length, :, :, :]

        return output

    def forward_encoder(self, conv, traj_gru, input):
        batch_size, sequence, input_channel, height, width = input.size()
        x = torch.reshape(input, (-1, input_channel, height, width))
        x = conv(x)
        x = torch.reshape(x, (batch_size, sequence, x.size(1), x.size(2), x.size(3)))
        output, state = traj_gru(x, None)
        return output, state

    def forward_forecaster(self, up_conv, traj_gru, input, hidden_state):
        x, state = traj_gru(input, hidden_state)
        batch_size, sequence, input_channel, height, width = x.size()
        x = torch.reshape(x, (-1, input_channel, height, width))
        x = up_conv(x)
        output = torch.reshape(x, (batch_size, sequence, x.size(1), x.size(2), x.size(3)))
        return output

### stepped down version

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import Conv, TrajGRU_cell, ConvTranspose

class TrajGRU(nn.Module):

    def __init__(self, device, input_length=4, output_length=3):
        super(TrajGRU, self).__init__()

        self.sequence_length = input_length
        self.output_length = output_length
        self.device = device

        channels_input = 1

        # Adjust the convolutional and TrajGRU layers to the reduced spatial dimensions
        self.conv_1 = Conv(in_channels=channels_input, out_channels=8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.traj_gru_1 = TrajGRU_cell(input_size=(8, 32, 32), hidden_filters=64, L=13, sequence_length=input_length, device=self.device)
        self.conv_2 = Conv(in_channels=64, out_channels=192, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.traj_gru_2 = TrajGRU_cell(input_size=(192, 16, 16), hidden_filters=192, L=13, sequence_length=input_length, device=self.device)
        self.conv_3 = Conv(in_channels=192, out_channels=192, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.traj_gru_3 = TrajGRU_cell(input_size=(192, 8, 8), hidden_filters=192, L=9, sequence_length=input_length, device=self.device)

        # Upsample the spatial dimensions
        self.traj_gru_4 = TrajGRU_cell(input_size=(192, 8, 8), hidden_filters=192, L=9, sequence_length=input_length, device=self.device)
        self.conv_transpose_1 = ConvTranspose(in_channels=192, out_channels=192, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.traj_gru_5 = TrajGRU_cell(input_size=(192, 16, 16), hidden_filters=192, L=13, sequence_length=input_length, device=self.device)
        self.conv_transpose_2 = ConvTranspose(in_channels=192, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.traj_gru_6 = TrajGRU_cell(input_size=(64, 32, 32), hidden_filters=64, L=13, sequence_length=input_length, device=self.device)
        self.conv_transpose_3 = ConvTranspose(in_channels=64, out_channels=8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.out_conv = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x):

        # Encoder
        output, state_1 = self.forward_encoder(self.conv_1, self.traj_gru_1, x)
        output, state_2 = self.forward_encoder(self.conv_2, self.traj_gru_2, output)
        output, state_3 = self.forward_encoder(self.conv_3, self.traj_gru_3, output)

        # Forecaster
        output = self.forward_forecaster(self.conv_transpose_1, self.traj_gru_4, None, state_3)
        output = self.forward_forecaster(self.conv_transpose_2, self.traj_gru_5, output, state_2)
        output = self.forward_forecaster(self.conv_transpose_3, self.traj_gru_6, output, state_1)

        output = torch.reshape(output, (-1, output.size(2), output.size(3), output.size(4)))
        output = F.relu(self.out_conv(output))
        output = torch.reshape(output, (output.size(0) // self.sequence_length, self.sequence_length, output.size(1), output.size(2), output.size(3)))

        # Adjust sequence length from input_length to output_length
        output = output[:, :self.output_length, :, :, :]

        return output

    def forward_encoder(self, conv, traj_gru, input):
        batch_size, sequence, input_channel, height, width = input.size()
        x = torch.reshape(input, (-1, input_channel, height, width))
        x = conv(x)
        x = torch.reshape(x, (batch_size, sequence, x.size(1), x.size(2), x.size(3)))
        output, state = traj_gru(x, None)
        return output, state

    def forward_forecaster(self, up_conv, traj_gru, input, hidden_state):
        x, state = traj_gru(input, hidden_state)
        batch_size, sequence, input_channel, height, width = x.size()
        x = torch.reshape(x, (-1, input_channel, height, width))
        x = up_conv(x)
        output = torch.reshape(x, (batch_size, sequence, x.size(1), x.size(2), x.size(3)))
        return output
