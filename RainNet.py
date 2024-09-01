#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 22:30:33 2024

@author: avijitmajhi

RainNet Model Training
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
import utility
import RainNet_model
from tqdm import tqdm  # For progress bar
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard

# Get the current working directory
current_dir = os.path.abspath(os.getcwd())

# Define directories based on the current working directory
data_dir = os.path.join(current_dir, "radar_data_unica_2018_2023")
excel_file = os.path.join(current_dir, "image_isw_scores.xlsx")
output_dir = os.path.join(current_dir, 'RainNet')
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
        images = np.zeros((self.steps, 256, 256), dtype=np.float32)
        
        for i in range(-15, 16, 5):
            time_step = current_time + pd.Timedelta(minutes=i)
            filename = utility.fname2dt(time_step, inverse=True)
            file_path = os.path.join(self.base_dir, filename)
            
            if os.path.exists(file_path):
                image = utility.read_image(file_path)
                image = utility.normalize(image, inverse=False)
                images[i // 5, :, :] = image  # Ensure image is placed in the correct channel
                
        x = images[0:-3, :, :]
        y = images[-3:, :, :]
        
        return x, y

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = RainNet_model.RainNet(input_channels=4, mode="regression")

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

log_interval = 10
epochs = 4

output_file = os.path.join(output_dir, 'rainet-out.pth')

def train(model, train_loader, val_loader, epochs, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = RainNet_model.log_cosh_loss  # or you could use another loss function

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
        
        val_loss /= len(val_loader.dataset)
        valid_losses.append(val_loss)
        
        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, output_file)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f"Epoch {epoch+1}/{epochs}.. Train loss: {train_loss:.4f}.. Val loss: {val_loss:.4f}")

# Call the train function
train(model, train_loader, valid_loader, epochs)

torch.cuda.empty_cache()
print(f'Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB, Max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB')

# Close the log file
log_file.close()

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, epochs+1), valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
plt.show()
