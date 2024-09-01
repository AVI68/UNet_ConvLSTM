# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:14:20 2024

@author: Avijit Majhi

RainPredNet Training
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as datetime
import utility
import nowcast_model
from tqdm import tqdm  # For progress bar
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard

# Get the current working directory
current_dir = os.path.abspath(os.getcwd())

# Define directories based on the current working directory
data_dir = os.path.join(current_dir, "radar_data_unica_2018_2023")
excel_file = os.path.join(current_dir, "image_isw_scores.xlsx")
output_dir = os.path.join(current_dir, 'RainPredNet')
os.makedirs(output_dir, exist_ok=True)
torch.cuda.empty_cache()


# Load the data
df = pd.read_excel(excel_file)


class RadarDataset(Dataset):
    """Custom Radar dataset for loading and processing radar images."""

    def __init__(self, times, base_dir, steps=31):
        self.times = times
        self.base_dir = base_dir
        self.steps = steps

    def __len__(self):
        return len(self.times) - self.steps

    def __getitem__(self, idx):
        current_time = self.times.iloc[idx]
        images = np.zeros((self.steps, 1, 256, 256), dtype=np.float32)
        
        for i in range(-15, 16):
            time_step = current_time + pd.Timedelta(minutes=i)
            filename = utility.fname2dt(time_step, inverse=True)
            file_path = os.path.join(self.base_dir, filename)
            
            if os.path.exists(file_path):
                image = utility.read_image(file_path)
                image = utility.normalize(image, inverse=False)
                images[i, 0, :, :] = image  # Ensure image is placed in the correct channel
        
        return images

BATCH_SIZE = 2

# Assuming the first column contains the datetime
times = pd.to_datetime(df.iloc[:, 0])

# Filter data for training years: 2018, 2020, 2021, 2022, and 2023
train_times = times[times.dt.year.isin([2018, 2020, 2021, 2022, 2023])]

# Filter data for validation: January to September 2019
valid_times = times[(times.dt.year == 2019) & (times.dt.month <= 9)]

# Filter data for testing: October to December 2019
test_times = times[(times.dt.year == 2019) & (times.dt.month >= 10)]

train_dataset = RadarDataset(train_times, data_dir)
valid_dataset = RadarDataset(valid_times, data_dir)
test_dataset =  RadarDataset(test_times, data_dir)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# # plot one batch of events (31 frames)
# dataiter = iter(test_loader)
# images = next(dataiter)  # Use the next() function instead of dataiter.next()
# plt.figure(figsize=(48, 4))
# bn = 0  # batch number

# # Plot the color map label on the first subplot (i+1)
# plt.subplot(2, 16, 32)
# plt.axis('off')  # Remove axis for clarity
# plt.imshow([[0,1]], cmap='turbo', aspect='auto')
# plt.colorbar(orientation='horizontal')
# plt.title('Color Map')

# # Plot the images from i+1 onwards
# for i in range(31):
#     plt.subplot(2, 16, i + 1)
#     plt.tight_layout()
#     plt.imshow(images[bn, i], cmap='turbo', interpolation='none')
# plt.show()


## Prednet model training 


# Initialize the model
A_channels = (1, 32, 64, 128)
R_channels = (1, 32, 64, 128)
model = nowcast_model.PredNet(R_channels, A_channels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize optimizer and scaler
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# Initialize TensorBoard writer
writer = SummaryWriter(os.path.join(output_dir, 'runs', 'experiment_1'))

# Initialize logging to file
log_file = open(os.path.join(output_dir, "training_log.txt"), "a")

train_losses = []
valid_losses = []



def train(epoch, log_interval):
    model.train()
    loss_log = 0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch}") as pbar:  # Progress bar
        for batch_idx, frames in enumerate(train_loader):
            frames = frames.to(device)
            optimizer.zero_grad()

            errors = model(frames, time_steps=16, forecast_steps=15, mode='error')
            loss = torch.mean(errors)
            loss.backward()
            optimizer.step()

            loss_log += loss.item()

            if (batch_idx + 1) % log_interval == 0:
                log_message = (f'Train Epoch: {epoch} [{(batch_idx + 1) * len(frames)}/{len(train_loader) * BATCH_SIZE} '
                               f'({100. * (batch_idx + 1) / len(train_loader):.0f}%)]\tLoss: {loss_log / log_interval:.6f}')
                print(log_message)
                log_file.write(log_message + "\n")  # Log to file
                writer.add_scalar('Training Loss', loss_log / log_interval, epoch * len(train_loader) + batch_idx)  # Log to TensorBoard
                pbar.set_postfix({"Loss": loss_log / log_interval})
                pbar.update(log_interval)
                loss_log = 0
    
    train_losses.append(loss_log / len(train_loader))

def valid():
    model.eval()
    loss_log = 0
    with torch.no_grad():
        with tqdm(total=len(valid_loader), desc="Validating") as pbar:  # Progress bar
            for batch_idx, frames in enumerate(valid_loader):
                frames = frames.to(device)
                
                errors = model(frames, time_steps=16, forecast_steps=15, mode='error')
                loss = torch.mean(errors)

                loss_log += loss.item()
                pbar.update(1)
        
        loss_log /= len(valid_loader)
        valid_losses.append(loss_log)
        print(f'\nValidation loss: {loss_log:.6f}\n')
        log_file.write(f'Validation loss: {loss_log:.6f}\n')  # Log to file
        writer.add_scalar('Validation Loss', loss_log, epoch)

torch.cuda.empty_cache()
print(f'Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB, Max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB')

log_interval = 10
n_epochs = 4


output_file = os.path.join(output_dir, 'prednet-out.pth')

for epoch in range(1, n_epochs + 1):
    train(epoch, log_interval)
    valid()
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_file)

torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, output_file)
# Close the log file
log_file.close()



plt.figure(figsize=(10, 5))
plt.plot(range(1, n_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, n_epochs + 1), valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
plt.show()
