#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 14:05:27 2024

@author: avijitmajhi
"""
from models.u_net import UNet
from models.naive_cnn import cnn_2D
from models.traj_gru import TrajGRU
from models.conv_gru import ConvGRU
import torch
import argparse
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utility
from tqdm import tqdm  # For progress bar
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard

class RadarDataset(Dataset):
    """Custom Radar dataset for loading and processing radar images."""
    def __init__(self, times, base_dir, input_steps=16, output_steps=15, recurrent_nn=False):
        self.times = times
        self.base_dir = base_dir
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.recurrent_nn = recurrent_nn

    def __len__(self):
        return len(self.times) - (self.input_steps + self.output_steps)

    def __getitem__(self, idx):
        current_time = self.times.iloc[idx]
        images = np.zeros((self.input_steps + self.output_steps, 256, 256), dtype=np.float32)
        
        for i in range(-(self.input_steps - 1), (self.output_steps + 1), 1):
            time_step = current_time + pd.Timedelta(minutes=i * 5)
            filename = utility.fname2dt(time_step, inverse=True)
            file_path = os.path.join(self.base_dir, filename)
            
            if os.path.exists(file_path):
                image = utility.read_image(file_path)
                image = utility.normalize(image, inverse=False)
                images[i + (self.input_steps - 1), :, :] = image  # Adjust index for correct placement
                
        if self.recurrent_nn:
            # For recurrent models, we add an extra dimension for the channel (which is 1 for rain maps)
            x = images[:self.input_steps, :, :].reshape(self.input_steps, 1, 256, 256)
            y = images[self.input_steps:, :, :].reshape(self.output_steps, 1, 256, 256)
        else:
            # For non-recurrent models, the input is typically a sequence of images without an additional channel dimension
            x = images[:self.input_steps, :, :]
            y = images[self.input_steps:, :, :]
        
        # Convert numpy arrays to PyTorch tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        return x, y


def train_network(network, train_loader, valid_loader, loss_type, epochs, batch_size, device, log_dir, print_metric_logs=False):

    writer = SummaryWriter(log_dir)

    n_examples_train = len(train_loader.dataset)
    n_examples_valid = len(valid_loader.dataset)

    lr = 1e-5
    wd = 0.1
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    info = f'''Starting training:
        Epochs:                {epochs},
        Learning rate:         {lr},
        Batch size:            {batch_size},
        Weight decay:          {wd},
        Number batch train :   {len(train_loader)},
        Number batch val :     {len(valid_loader)},
        Scheduler :            Gamma 0.1 epochs 30, 60
    '''
    writer.add_text('Description', info)

    for epoch in range(epochs):
        network.train()
        training_loss = 0.0
        validation_loss = 0.0

        loop = tqdm(train_loader)
        loop.set_description(f"Epoch {epoch+1}/{epochs}")

        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            optimizer.zero_grad()
            outputs = network(inputs)
            if loss_type == 'CB_loss':
                loss = utility.CB_loss(outputs, targets)
            elif loss_type == 'MCS_loss':
                loss = utility.MCS_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() / n_examples_train

            loop.set_postfix({'Train Loss': training_loss})

        scheduler.step()

        network.eval()

        for inputs, targets in valid_loader:
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            outputs = network(inputs)
            if loss_type == 'CB_loss':
                loss = utility.CB_loss(outputs, targets)
            elif loss_type == 'MCS_loss':
                loss = utility.MCS_loss(outputs, targets)
            validation_loss += loss.item() / n_examples_valid

        writer.add_scalar('Loss/train', training_loss, epoch)
        writer.add_scalar('Loss/test', validation_loss, epoch)
        print(f"[Validation] Loss: {validation_loss:.2f}")

        torch.save(network, log_dir + f'/model_{epoch+1}.pth')

if __name__ == '__main__':
    # Load the data
    excel_file = "/content/UNet_ConvLSTM/image_isw_scores.xlsx"
    df = pd.read_excel(excel_file)
    # Assuming the first column contains the datetime
    times = pd.to_datetime(df.iloc[:, 0])

    # Filter data for training years: 2018, 2020, 2021, 2022, and 2023
    train_times = times[times.dt.year.isin([2018, 2020, 2021, 2022, 2023])]

    # Filter data for validation: January to September 2019
    valid_times = times[(times.dt.year == 2019) & (times.dt.month <= 9)]

    # Filter data for testing: October to December 2019
    test_times = times[(times.dt.year == 2019) & (times.dt.month >= 10)]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=4, type=int, help="The number of epochs used to train the network")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--input_length', type=int, default=16, help="The number of time steps of a sequence as input of the NN")
    parser.add_argument('--output_length', type=int, default=15, help="The number of time steps predicted by the NN")
    parser.add_argument('--print_metric_logs', action='store_true', help='If we want to print the metrics score while training')
    parser.add_argument('--network', choices=['TrajGRU', 'ConvGRU', 'CNN2D', 'UNet'])
    parser.add_argument('--loss_type', choices=['CB_loss', 'MCS_loss'], required=True)
    args = parser.parse_args()
    
    # Log directory
    log_dir = f"/content/drive/MyDrive/run/network_{args.network}_epochs_{args.epochs}_batch_size_{args.batch_size}_IL_{args.input_length}_OL_{args.output_length}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    if args.network == 'ConvGRU':
        recurrent_nn = True
        network = ConvGRU(device=device, input_length=args.input_length, output_length=args.output_length)
    elif args.network == 'TrajGRU':
        recurrent_nn = True
        network = TrajGRU(device=device, input_length=args.input_length, output_length=args.output_length)
    elif args.network == 'CNN2D':
        recurrent_nn = False
        network = cnn_2D(input_length=args.input_length, output_length=args.output_length, filter_number=64)
    elif args.network == 'UNet':
        recurrent_nn = False
        network = UNet(input_length=args.input_length, output_length=args.output_length, filter_number=64)
       
    network.to(device=device)
    
    # Adjusted base directory to point to the unzipped folder
    data_dir = "/content/radar_data_unica_2018_2023_sorted" 

    train_dataset = RadarDataset(train_times, data_dir, input_steps=args.input_length, output_steps=args.output_length, recurrent_nn=recurrent_nn)
    valid_dataset = RadarDataset(valid_times, data_dir, input_steps=args.input_length, output_steps=args.output_length, recurrent_nn=recurrent_nn)
    test_dataset = RadarDataset(test_times, data_dir, input_steps=args.input_length, output_steps=args.output_length, recurrent_nn=recurrent_nn)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    train_network(network, train_loader, valid_loader,
                  args.loss_type,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  device=device,
                  log_dir=log_dir,
                  print_metric_logs=args.print_metric_logs)

