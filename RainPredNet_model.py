# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 19:46:18 2024
@author: Avijit Majhi

RainPredNet Model
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
    def forward(self, input_tensor, cur_state):
        device = self.conv.weight.device
        input_tensor = input_tensor.to(device)
        h_cur, c_cur = cur_state
        h_cur = h_cur.to(device)
        c_cur = c_cur.to(device)
        combined = torch.cat([input_tensor, h_cur], dim=1)  # Concatenate along the channel dimension
        combined_conv = self.conv(combined)  # Convolution operation
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)
    def reset_parameters(self):
        self.conv.reset_parameters()
    def init_hidden(self, batch_size, width, height):
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, width, height, device=device),
                torch.zeros(batch_size, self.hidden_dim, width, height, device=device))
class SatLU(nn.Module):
    def __init__(self, lower=0, upper=1, inplace=False):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace
    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper, self.inplace)
    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
               + 'min_val=' + str(self.lower) \
               + ', max_val=' + str(self.upper) \
               + inplace_str + ')'
class PredNet(nn.Module):
    def __init__(self, R_channels, A_channels):
        # Define some constants
        KERNEL_SIZE = 3
        PADDING = KERNEL_SIZE//2 
        super(PredNet, self).__init__()
        self.r_channels = R_channels + (0,)  # for convenience
        self.a_channels = A_channels
        self.n_layers = len(R_channels)
        for i in range(self.n_layers):
            cell = ConvLSTMCell(2 * self.a_channels[i] + self.r_channels[i + 1],
                                self.r_channels[i], KERNEL_SIZE, True)
            setattr(self, 'cell{}'.format(i), cell)
        for i in range(self.n_layers):
            conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], KERNEL_SIZE, padding=PADDING), nn.ReLU())
            if i == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(nn.Conv2d(2 * self.a_channels[l], self.a_channels[l + 1], KERNEL_SIZE, padding=PADDING), self.maxpool)
            setattr(self, 'update_A{}'.format(l), update_A)
        self.reset_parameters()
    def reset_parameters(self):
        for l in range(self.n_layers):
            cell = getattr(self, 'cell{}'.format(l))
            cell.reset_parameters()
    def forward(self, input, time_steps=1, forecast_steps=0, mode='error'):
        device = input.device  # Ensure everything happens on the device of the input tensor
        R_seq = [None] * self.n_layers
        H_seq = [None] * self.n_layers
        E_seq = [None] * self.n_layers
        batch_size, input_steps, channels, width, height = input.size()
        # print(f"Input shape: {input.shape}")
        w = width
        h = height
        for l in range(self.n_layers):
            E_seq[l] = torch.randn(batch_size, 2 * self.a_channels[l], w, h, device=device)
            R_seq[l] = torch.randn(batch_size, self.r_channels[l], w, h, device=device)
            # print(f"Initial E_seq[{l}] shape: {E_seq[l].shape}")
            # print(f"Initial R_seq[{l}] shape: {R_seq[l].shape}")
            w = w // 2
            h = h // 2
        if mode == 'error':
            total_error = []
        else:
            output = []
        for t in range(time_steps + forecast_steps):
            if t < input_steps:
                frame_input = input[:, t].to(device)  # Ensure frame_input is on the correct device
            else:
                frame_input = None
            # print(f"Time step {t}, frame_input shape: {frame_input.shape if frame_input is not None else 'None'}")
            for l in reversed(range(self.n_layers)):
                cell = getattr(self, 'cell{}'.format(l))
                if t == 0:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = (R, R)
                else:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = H_seq[l]
                if l == self.n_layers - 1:
                    R, hx = cell(E, hx)
                else:
                    E = E.to(device)
                    interpolated_R = F.interpolate(R_seq[l + 1], scale_factor=2, mode='bilinear', align_corners=False).to(device)
                    tmp = torch.cat((E, interpolated_R), 1)
                    # print(f"Layer {l}, tmp shape after interpolation and concat: {tmp.shape}")
                    R, hx = cell(tmp, hx)
                R_seq[l] = R.to(device)
                H_seq[l] = hx
                # print(f"Layer {l}, R_seq[{l}] shape: {R_seq[l].shape}")
            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l)).to(device)  # Ensure the convolution layer is on the correct device
                A_hat = conv(R_seq[l])  # Convolution operation
                if l == 0:
                    frame_prediction = A_hat
                    if t < time_steps:
                        A = frame_input  # Use the frame input as A
                    else:
                        A = frame_prediction  # Use the predicted frame as A
                A = A.to(device)
                A_hat = A_hat.to(device)
                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E = torch.cat([pos, neg], 1)
                E_seq[l] = E.to(device)
                # print(f"Layer {l}, E_seq[{l}] shape after concat: {E_seq[l].shape}")
                if l < self.n_layers - 1:
                    update_A = getattr(self, 'update_A{}'.format(l)).to(device)  # Ensure the update_A layer is on the correct device
                    A = update_A(E)
                    # print(f"Layer {l}, A shape after update_A: {A.shape}")
            if mode == 'error':
                if frame_input is not None:
                    error = torch.mean((frame_input - frame_prediction) ** 2)
                    total_error.append(error.to(device))
            else:
                output.append(frame_prediction.to(device))
        if mode == 'error':
            return torch.stack(total_error, 0).to(device)
        else:
            return torch.stack(output, 1).to(device)