#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 01:30:48 2024

@author: avijitmajhi
"""
# import tensorflow as tf
# import os
# import numpy as np
# import pandas as pd
# import utility_tf as utility
# from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
# from tensorflow.keras.optimizers import Adam
# from models.UNet_ConvLSTM import unet_convlstm_reg
# import gc  # Garbage collector

# # Paths
# data_dir = "C:/Users/Utente/radar_data_unica_2018_2023_sorted"
# excel_file = "C:/Users/Utente/UNet_ConvLSTM/image_isw_scores.xlsx"
# output_dir = "C:/Users/Utente/UNet_ConvLSTM/"
# checkpoint_dir = os.path.join(output_dir, 'checkpoints')

# # Create necessary directories
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(checkpoint_dir, exist_ok=True)

# # Load the data
# df = pd.read_excel(excel_file)

# class RadarDatasetTF:
#     """Custom Radar dataset for loading and processing radar images for TensorFlow."""
#     def __init__(self, times, base_dir, input_steps=16, output_steps=15):
#         self.times = times
#         self.base_dir = base_dir
#         self.input_steps = input_steps
#         self.output_steps = output_steps

#     def __len__(self):
#         return len(self.times) - (self.input_steps + self.output_steps)

#     def _load_sample(self, idx):
#         current_time = self.times.iloc[idx]
#         images = np.zeros((self.input_steps + self.output_steps, 256, 256), dtype=np.float32)
        
#         for i in range(-(self.input_steps - 1), (self.output_steps + 1), 1):
#             time_step = current_time + pd.Timedelta(minutes=i)
#             filename = utility.fname2dt(time_step, inverse=True)
#             file_path = os.path.join(self.base_dir, filename)
            
#             if os.path.exists(file_path):
#                 image = utility.read_image(file_path)
#                 image = utility.normalize(image, inverse=False)
#                 images[i + (self.input_steps - 1), :, :] = image  # Adjust index for correct placement
                
#         x = images[:self.input_steps, :, :].reshape(self.input_steps, 256, 256, 1)
#         y = images[self.input_steps:, :, :].reshape(self.output_steps, 256, 256, 1)
        
#         return x, y

#     def to_tf_dataset(self, batch_size=2, shuffle=True):
#         dataset = tf.data.Dataset.from_tensor_slices(np.arange(len(self)))
#         if shuffle:
#             dataset = dataset.shuffle(buffer_size=len(self))
#         dataset = dataset.map(lambda idx: tf.numpy_function(
#             func=self._load_sample, inp=[idx], Tout=(tf.float32, tf.float32)),
#             num_parallel_calls=tf.data.AUTOTUNE)
#         dataset = dataset.batch(batch_size)
#         dataset = dataset.prefetch(tf.data.AUTOTUNE)
#         return dataset

# # Prepare datasets
# times = pd.to_datetime(df.iloc[:, 0])
# train_times = times[times.dt.year.isin([2018, 2020, 2021, 2022, 2023])]
# valid_times = times[(times.dt.year == 2019) & (times.dt.month <= 9)]
# test_times = times[(times.dt.year == 2019) & (times.dt.month >= 10)]

# train_dataset = RadarDatasetTF(train_times, data_dir)
# valid_dataset = RadarDatasetTF(valid_times, data_dir)
# test_dataset = RadarDatasetTF(test_times, data_dir)

# BATCH_SIZE =2
# train_loader = train_dataset.to_tf_dataset(batch_size=BATCH_SIZE)
# valid_loader = valid_dataset.to_tf_dataset(batch_size=BATCH_SIZE)
# test_loader = test_dataset.to_tf_dataset(batch_size=BATCH_SIZE)

# def train_model(loss_function_name):
#     """
#     Train the model using the specified loss function.
    
#     Parameters:
#         loss_function_name (str): The name of the loss function to use ('MCS_loss' or 'CB_loss').
#     """
#     # Select the loss function based on the parameter
#     if loss_function_name == 'MCS_loss':
#         loss_function = utility.MCS_loss
#     elif loss_function_name == 'CB_loss':
#         loss_function = utility.CB_loss
#     elif loss_function_name == 'Bmae_loss':
#         loss_function = utility.Bmae_loss
#     elif loss_function_name == 'Bmse_loss':
#         loss_function = utility.Bmse_loss
#     elif loss_function_name == 'mse_loss':
#         loss_function = 'mse'
#     else:
#         raise ValueError("Invalid loss function name. Choose 'MCS_loss','CB_loss','Bmae_loss','Bmse_loss','mse_loss'.")
    
#     # Define model
#     with tf.device('/GPU:0'):
#         model = unet_convlstm_reg(input_shape=(16, 256, 256, 1), num_filters_base=16, dropout_rate=0.2, seq_len=15)
#         model.compile(optimizer=Adam(learning_rate=0.0001), loss=loss_function)
    
#     # Callbacks
#     tensorboard_callback = TensorBoard(log_dir=os.path.join(log_dir, 'experiment_1', loss_function_name))
#     checkpoint_callback = ModelCheckpoint(
#         filepath=os.path.join(checkpoint_dir, f'model_{loss_function_name}_epoch_{{epoch:02d}}.h5'),
#         save_weights_only=False,
#         save_freq='epoch'
#     )
#     csv_logger = CSVLogger(os.path.join(log_dir, f'training_log_{loss_function_name}.csv'))

#     # Training
#     n_epochs = 4
#     with tf.device('/GPU:0'):
#         history = model.fit(
#             train_loader,
#             epochs=n_epochs,
#             validation_data=valid_loader,
#             callbacks=[tensorboard_callback, checkpoint_callback, csv_logger]
#         )

#     # Clear memory after training
#     tf.keras.backend.clear_session()
#     gc.collect()

#     # Save the final model
#     model.save(os.path.join(output_dir, f'unet_convlstm_{loss_function_name}-final.h5'))

#     # Evaluate on the test set
#     with tf.device('/GPU:0'):
#         test_loss = model.evaluate(test_loader)
#         print(f"Test loss with {loss_function_name}: {test_loss:.6f}")

#     # Clear memory after evaluation
#     tf.keras.backend.clear_session()
#     gc.collect()

# if __name__ == '__main__':
#     # Prompt the user to choose the loss function
#     print("Choose the loss function to train the model:")
#     print("1: MCS_loss")
#     print("2: CB_loss")
#     print("3: Bmae_loss")
#     print("4: Bmse_loss")
#     print("5: mse_loss")
#     loss_choice = input("Enter the number corresponding to the loss function: ")

#     if loss_choice == '1':
#         loss_function_name = 'MCS_loss'
#     elif loss_choice == '2':
#         loss_function_name = 'CB_loss'
#     elif loss_choice == '3':
#         loss_function_name = 'Bmae_loss'
#     elif loss_choice == '4':
#         loss_function_name = 'Bmse_loss'
#     elif loss_choice == '5':
#         loss_function_name = 'mse_loss'
#     else:
#         raise ValueError("Invalid choice! Please run the script again and choose a valid option.")

#     # Define the log directory based on user choice
#     log_dir = f"C:/Users/Utente/UNet_ConvLSTM/run/network_unet_convlstm_epochs_4_batch_size_{BATCH_SIZE}_IL_16_OL_15_loss_{loss_function_name}"

#     # Create necessary directories
#     os.makedirs(log_dir, exist_ok=True)

#     # Train the model with the chosen loss function
#     train_model(loss_function_name)

import tensorflow as tf
import os
import numpy as np
import pandas as pd
import utility_tf as utility
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from models.UNet_ConvLSTM import unet_convlstm_reg
import gc  # Garbage collector

# Paths
data_dir = "C:/Users/Utente/radar_data_unica_2018_2023_sorted"
excel_file = "C:/Users/Utente/UNet_ConvLSTM/image_isw_scores.xlsx"
output_dir = "C:/Users/Utente/UNet_ConvLSTM/"
checkpoint_dir = os.path.join(output_dir, 'checkpoints')

# Create necessary directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Load the data
df = pd.read_excel(excel_file)

class RadarDatasetTF:
    """Custom Radar dataset for loading and processing radar images for TensorFlow."""
    def __init__(self, times, base_dir, input_steps=4, output_steps=3):
        self.times = times
        self.base_dir = base_dir
        self.input_steps = input_steps
        self.output_steps = output_steps

    def __len__(self):
        return len(self.times) - (self.input_steps + self.output_steps)

    def _load_sample(self, idx):
        current_time = self.times.iloc[idx]
        
        # Define the time intervals in minutes for both input and output
        combined_intervals = [-15, -5, 5, 15, 0, 5, 10]
        total_steps = self.input_steps + self.output_steps
        
        # Create an array for the images
        images = np.zeros((total_steps, 64, 64), dtype=np.float32)
        
        # Load images at the specified intervals
        for i, offset in enumerate(combined_intervals):
            time_step = current_time + pd.Timedelta(minutes=offset)
            filename = utility.fname2dt(time_step, inverse=True)
            file_path = os.path.join(self.base_dir, filename)
            
            if os.path.exists(file_path):
                image = utility.read_image(file_path)
                image = utility.normalize(image, inverse=False)
                images[i, :, :] = image
        
        # Split the loaded images into input (x) and output (y)
        x = images[:self.input_steps, :, :].reshape(self.input_steps, 64, 64, 1)
        y = images[self.input_steps:, :, :].reshape(self.output_steps, 64, 64, 1)
        
        return x, y

    def to_tf_dataset(self, batch_size=32, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(len(self)))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self))
        dataset = dataset.map(lambda idx: tf.numpy_function(
            func=self._load_sample, inp=[idx], Tout=(tf.float32, tf.float32)),
            num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

# Prepare datasets
times = pd.to_datetime(df.iloc[:, 0])
train_times = times[times.dt.year.isin([2018, 2020, 2021, 2022, 2023])]
valid_times = times[(times.dt.year == 2019) & (times.dt.month <= 9)]
test_times = times[(times.dt.year == 2019) & (times.dt.month >= 10)]

train_dataset = RadarDatasetTF(train_times, data_dir)
valid_dataset = RadarDatasetTF(valid_times, data_dir)
test_dataset = RadarDatasetTF(test_times, data_dir)

BATCH_SIZE = 32
train_loader = train_dataset.to_tf_dataset(batch_size=BATCH_SIZE)
valid_loader = valid_dataset.to_tf_dataset(batch_size=BATCH_SIZE)
test_loader = test_dataset.to_tf_dataset(batch_size=BATCH_SIZE)

def train_model(loss_function_name):
    """
    Train the model using the specified loss function.
    
    Parameters:
        loss_function_name (str): The name of the loss function to use ('MCS_loss', 'CB_loss', 'Bmae_loss', 'Bmse_loss', 'mse_loss', 'mae_loss').
    """
    # Select the loss function based on the parameter
    if loss_function_name == 'MCS_loss':
        loss_function = utility.MCS_loss
    elif loss_function_name == 'CB_loss':
        loss_function = utility.CB_loss
    elif loss_function_name == 'Bmae_loss':
        loss_function = utility.Bmae_loss
    elif loss_function_name == 'Bmse_loss':
        loss_function = utility.Bmse_loss
    elif loss_function_name == 'mse_loss':
        loss_function = 'mse'
    elif loss_function_name == 'mae_loss':
        loss_function = 'mae'
    else:
        raise ValueError("Invalid loss function name. Choose 'MCS_loss','CB_loss','Bmae_loss','Bmse_loss','mse_loss', or 'mae_loss'.")
    
    # Define model
    with tf.device('/GPU:0'):
        model = unet_convlstm_reg(input_shape=(4, 64, 64, 1), num_filters_base=16, dropout_rate=0.2, seq_len=3)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss=loss_function)
    
    # Callbacks
    tensorboard_callback = TensorBoard(log_dir=os.path.join(log_dir, 'experiment_1', loss_function_name))
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, f'model_{loss_function_name}_epoch_{{epoch:02d}}.h5'),
        save_weights_only=False,
        save_freq='epoch'
    )
    csv_logger = CSVLogger(os.path.join(log_dir, f'training_log_{loss_function_name}.csv'))

    # Training
    n_epochs = 4
    with tf.device('/GPU:0'):
        history = model.fit(
            train_loader,
            epochs=n_epochs,
            validation_data=valid_loader,
            callbacks=[tensorboard_callback, checkpoint_callback, csv_logger]
        )

    # Clear memory after training
    tf.keras.backend.clear_session()
    gc.collect()

    # Save the final model
    model.save(os.path.join(output_dir, f'unet_convlstm_{loss_function_name}-final.h5'))

    # Evaluate on the test set
    with tf.device('/GPU:0'):
        test_loss = model.evaluate(test_loader)
        print(f"Test loss with {loss_function_name}: {test_loss:.6f}")

    # Clear memory after evaluation
    tf.keras.backend.clear_session()
    gc.collect()

if __name__ == '__main__':
    # Prompt the user to choose the loss function
    print("Choose the loss function to train the model:")
    print("1: MCS_loss")
    print("2: CB_loss")
    print("3: Bmae_loss")
    print("4: Bmse_loss")
    print("5: mse_loss")
    print("6: mae_loss")
    loss_choice = input("Enter the number corresponding to the loss function: ")

    if loss_choice == '1':
        loss_function_name = 'MCS_loss'
    elif loss_choice == '2':
        loss_function_name = 'CB_loss'
    elif loss_choice == '3':
        loss_function_name = 'Bmae_loss'
    elif loss_choice == '4':
        loss_function_name = 'Bmse_loss'
    elif loss_choice == '5':
        loss_function_name = 'mse_loss'
    elif loss_choice == '6':
        loss_function_name = 'mae_loss'
    else:
        raise ValueError("Invalid choice! Please run the script again and choose a valid option.")

    # Define the log directory based on user choice
    log_dir = f"C:/Users/Utente/UNet_ConvLSTM/run/network_unet_convlstm_epochs_4_batch_size_{BATCH_SIZE}_IL_4_OL_3_loss_{loss_function_name}"

    # Create necessary directories
    os.makedirs(log_dir, exist_ok=True)

    # Train the model with the chosen loss function
    train_model(loss_function_name)
