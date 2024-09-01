#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 04:07:17 2024

@author: avijitmajhi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTM2D(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1, batch_first=True, bias=True, return_sequences=True):
        super(ConvLSTM2D, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_sequences = return_sequences

        self.convlstm = nn.LSTM(input_size=input_dim,
                                hidden_size=hidden_dim,
                                num_layers=num_layers,
                                batch_first=batch_first,
                                bias=bias)

        self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        h, c = self.init_hidden(batch_size, self.hidden_dim, (height, width))

        outputs = []
        for t in range(seq_len):
            x_in = x[:, t, :, :, :].view(batch_size, channels, height, width)
            x_in = self.conv(x_in)
            x_in = x_in.view(batch_size, -1).unsqueeze(1)  # Preparing for LSTM
            x_in, (h, c) = self.convlstm(x_in, (h, c))
            x_in = x_in.squeeze(1).view(batch_size, self.hidden_dim, height, width)
            outputs.append(x_in)

        outputs = torch.stack(outputs, dim=1)
        if not self.return_sequences:
            outputs = outputs[:, -1, :, :, :]

        return outputs

    def init_hidden(self, batch_size, hidden_dim, spatial_dims):
        h = torch.zeros(self.num_layers, batch_size, hidden_dim, *spatial_dims)
        c = torch.zeros(self.num_layers, batch_size, hidden_dim, *spatial_dims)
        return h, c


class UNetConvLSTM(nn.Module):
    def __init__(self, input_shape=(16, 1, 256, 256), num_filters_base=4, dropout_rate=0.2, seq_len=15):
        super(UNetConvLSTM, self).__init__()

        self.seq_len = seq_len
        self.num_filters_base = num_filters_base

        self.conv_lstm1 = ConvLSTM2D(input_dim=input_shape[1], hidden_dim=num_filters_base, kernel_size=3, return_sequences=True)
        self.pool1 = nn.MaxPool3d((1, 2, 2), padding=(0, 1, 1))
        self.batch_norm1 = nn.BatchNorm3d(num_filters_base)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv_lstm2 = ConvLSTM2D(input_dim=num_filters_base, hidden_dim=2 * num_filters_base, kernel_size=3, return_sequences=True)
        self.pool2 = nn.MaxPool3d((1, 2, 2), padding=(0, 1, 1))
        self.batch_norm2 = nn.BatchNorm3d(2 * num_filters_base)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv_lstm3 = ConvLSTM2D(input_dim=2 * num_filters_base, hidden_dim=4 * num_filters_base, kernel_size=3, return_sequences=True)
        self.pool3 = nn.MaxPool3d((1, 2, 2), padding=(0, 1, 1))
        self.batch_norm3 = nn.BatchNorm3d(4 * num_filters_base)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.conv_lstm4 = ConvLSTM2D(input_dim=4 * num_filters_base, hidden_dim=8 * num_filters_base, kernel_size=3, return_sequences=True)
        self.pool4 = nn.MaxPool3d((1, 2, 2), padding=(0, 1, 1))
        self.batch_norm4 = nn.BatchNorm3d(8 * num_filters_base)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.conv_lstm5 = ConvLSTM2D(input_dim=8 * num_filters_base, hidden_dim=8 * num_filters_base, kernel_size=3, return_sequences=False)

        self.deconv5 = nn.ConvTranspose3d(8 * num_filters_base, 8 * num_filters_base, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        self.batch_norm5 = nn.BatchNorm3d(8 * num_filters_base)
        self.dropout5 = nn.Dropout(dropout_rate)

        self.deconv6 = nn.ConvTranspose3d(12 * num_filters_base, 4 * num_filters_base, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        self.batch_norm6 = nn.BatchNorm3d(4 * num_filters_base)
        self.dropout6 = nn.Dropout(dropout_rate)

        self.deconv7 = nn.ConvTranspose3d(6 * num_filters_base, 2 * num_filters_base, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        self.batch_norm7 = nn.BatchNorm3d(2 * num_filters_base)
        self.dropout7 = nn.Dropout(dropout_rate)

        self.deconv8 = nn.ConvTranspose3d(3 * num_filters_base, num_filters_base, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        self.batch_norm8 = nn.BatchNorm3d(num_filters_base)
        self.dropout8 = nn.Dropout(dropout_rate)

        self.final_conv = nn.Conv3d(2 * num_filters_base, 1, kernel_size=1)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()

        x1 = self.conv_lstm1(x)
        x1 = self.pool1(x1)
        x1 = self.batch_norm1(x1)
        x1 = self.dropout1(x1)

        x2 = self.conv_lstm2(x1)
        x2 = self.pool2(x2)
        x2 = self.batch_norm2(x2)
        x2 = self.dropout2(x2)

        x3 = self.conv_lstm3(x2)
        x3 = self.pool3(x3)
        x3 = self.batch_norm3(x3)
        x3 = self.dropout3(x3)

        x4 = self.conv_lstm4(x3)
        x4 = self.pool4(x4)
        x4 = self.batch_norm4(x4)
        x4 = self.dropout4(x4)

        x5 = self.conv_lstm5(x4)
        x5 = self.deconv5(x5)
        x5 = self.batch_norm5(x5)
        x5 = self.dropout5(x5)

        x6 = self.deconv6(torch.cat([x5, x4], dim=1))
        x6 = self.batch_norm6(x6)
        x6 = self.dropout6(x6)

        x7 = self.deconv7(torch.cat([x6, x3], dim=1))
        x7 = self.batch_norm7(x7)
        x7 = self.dropout7(x7)

        x8 = self.deconv8(torch.cat([x7, x2], dim=1))
        x8 = self.batch_norm8(x8)
        x8 = self.dropout8(x8)

        output = self.final_conv(torch.cat([x8, x1], dim=1))
        output = output.squeeze(2)  # Remove the singleton depth dimension

        return output


