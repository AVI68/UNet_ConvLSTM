# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:33:40 2024

@author: Avijit Majhi
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2
import torch


def filelist(folder_path):
    """
    Lists all files in a given folder.
    """
    
    return [
        # Join the folder path with the file name to get the full path
        os.path.join(folder_path, f)
        # Iterate over all entries in the folder
        for f in os.listdir(folder_path) 
        # Include only files (not directories)
        # if os.path.isfile(os.path.join(folder_path, f))  
        if f.endswith('.png')
    ]

def read_image(file_path):
    """
    Read the image file.
    """
    try:
        img = Image.open(file_path)
        if img.mode != 'LA':
            raise ValueError("Image must have 2 channels (mask and intensity).")
        mask = np.asarray(img)[:, :, 1] / 255  # Normalize mask values to 0-1
        mask = mask.astype(np.uint8)  # Convert to 0 or 1 (uint8)
        intensity = np.asarray(img)[:, :, 0]
        masked_intensity = intensity * mask
        return masked_intensity
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
    except ValueError as e:
        print(f"Error: {e}")
    return None

def conversion(image, operation="DN_DBZ", inverse=False):
    image = image.astype(float)
    
    if not inverse:
        if operation == "DN_DBZ":
            # Convert from digital number (DN) to reflectivity in dBZ
            result = (image / 2.55) - 100 + 91.4
        elif operation == "DBZ_Z":
            # Convert from dBZ to Z
            result = 10.0 ** (image / 10.0)
        elif operation == "Z_R":
            # Convert from Z to rainfall rate R
            result = (image / 200) ** (1.0 / 1.6)
            result[result < 0.01] = 0
        else:
            raise ValueError("Invalid operation")
    else:
        if operation == "DN_DBZ":
            # Convert from dBZ to digital number (DN)
            result = np.round((image + 8.6) * 2.55)
            result[result <= 0] = 0
            result = result.astype(np.uint8)
        elif operation == "DBZ_Z":
            # Convert from Z to dBZ
            image[image <= 0] = 0.001
            result = 10.0 * np.log10(image)
        elif operation == "Z_R":
            # Convert from rainfall rate R to Z
            result = 200 * image ** 1.6
        else:
            raise ValueError("Invalid operation")
    return result

def normalize(image, inverse=False):
    min_val = 0
    max_val = 160

    if not inverse:
        # Clip the image values to ensure they stay within 0-160
        mapped_image = np.clip(image, min_val, max_val)
        mapped_image = mapped_image.astype(np.float32)  # Convert to float for further processing

        # Apply a median filter to reduce granular noise
        filtered_image = cv2.medianBlur(mapped_image, ksize=3)

        # Resize the filtered image to 256x256 using INTER_AREA interpolation (good for downscaling)
        resized_image = cv2.resize(filtered_image, (256, 256), interpolation=cv2.INTER_AREA)

        # Normalize the resized image to the 0-1 range based on the 0-160 scale
        standardized_image = resized_image / max_val

        return standardized_image

    else:
        # Reverse the normalization: scale back to the 0-160 range
        mapped_image = image * max_val
        mapped_image = np.clip(mapped_image, min_val, max_val)  # Clip values to stay within the 0-160 range
        mapped_image = mapped_image.astype(np.uint8)  # Convert to uint8 for image operations

        # Resize back to 256x256 
        resized_image = cv2.resize(mapped_image, (256, 256), interpolation=cv2.INTER_AREA)

        return resized_image


def fname2dt(value, inverse=False):
    """
    Convert between filename and datetime information.

    Parameters:
        value (str or datetime): The filename string or datetime object.
        inverse (bool): If False, extract datetime from filename.
                        If True, generate filename from datetime.

    Returns:
        datetime or str: Returns datetime object if inverse=False,
                         Returns filename string if inverse=True.
    """
        
    if not inverse:
        # Extract datetime from filename
        datetime_info = value.split(".")[0].split("\\")[-1]
        date_time_info = datetime.strptime(datetime_info, '%Y%m%d_%H%M')
        return date_time_info
    else:
        # Generate filename from datetime
        date_format = value.strftime('%Y%m%d_%H%M')
        filename = f"{date_format}.png"
        return filename

# Weights for different rainfall intensity
def weight(intensity):
  """
  Calculates the weight based on the rainfall intensity and pre-defined power values.
  Preserves NaN values in the output.
  """
  power = np.zeros_like(intensity)
  power[(~np.isnan(intensity)) & (intensity > 0) & (intensity <= 1)] = 1.0
  power[(~np.isnan(intensity)) & (intensity > 1) & (intensity <= 10)] = 1.5
  power[(~np.isnan(intensity)) & (intensity > 10)] = 2.0
  return np.power(intensity, power)
def isw_score(R):
    """
    Calculate the ISW score for a given intensity array.
    """
    # image_data, metadata = read_image(file_path)
    # reflectivity_dbz, metadata = dn_to_dbz(image_data, metadata)
    # intensity, metadata = calculate_rainfall_intensity(reflectivity_dbz, metadata)
    
    # Replace NaNs and values outside [0, 80] with 0 in intensity
    R = np.where((R >= 0) & (R <= 120), R, 0)
    
    weights = weight(R)
    isw_score = np.sum(weights * R) 
    return isw_score

def compute_weight_mask(target):
    """
    Compute a weight mask based on the target values.

    Thresholds: [0, 1, 5, 10, 20, 50]
    Weights: [1., 2., 5., 10., 50.]

    Parameters:
    - target: Tensor of target values.

    Returns:
    - mask: Tensor of weights based on the thresholds.
    """
    threshold = [0, 1, 5, 10, 20, 50]
    weights = [1., 2., 5., 10., 50.]

    # Assuming conversion functions are applied here
    threshol = conversion(conversion(conversion(np.array(threshold), "Z_R", inverse=True), "DBZ_Z", inverse=True), "DN_DBZ", inverse=True)
    threshol = threshol / 160
    # Ensure the weight mask and calculations are on the same device as the target tensor
    device = target.device

    # Convert numpy array to a PyTorch tensor and move it to the same device as target
    threshol = torch.tensor(threshol, dtype=torch.float32, device=device)

    

    # Initialize the mask with ones on the correct device
    mask = torch.ones_like(target, dtype=torch.double, device=device)

    # Apply weights based on thresholds
    for k in range(len(weights)):
        mask = torch.where(
            (threshol[k] <= target) & (target < threshol[k + 1]), 
            torch.tensor(weights[k], dtype=torch.double, device=device), 
            mask
        )

    return mask


def Bmse_loss(output, target):
    """
    Compute the weighted Mean Squared Error (MSE) loss.

    Parameters:
    - output: The predicted output tensor from the model.
    - target: The ground truth tensor.

    Returns:
    - loss: The weighted MSE loss as a single scalar tensor.
    """
    # Compute the weight mask based on the target
    weight_mask = compute_weight_mask(target)

    # Compute the squared difference between the output and target
    squared_diff = (output - target) ** 2

    # Apply the weight mask to the squared differences
    weighted_squared_diff = torch.multiply(weight_mask, squared_diff)

    # Sum up all the weighted squared differences to get the final loss
    loss = torch.sum(weighted_squared_diff)

    return loss

def Bmae_loss(output, target):
    """
    Compute the weighted Mean Absolute Error (MAE) loss.

    Parameters:
    - output: The predicted output tensor from the model.
    - target: The ground truth tensor.

    Returns:
    - loss: The weighted MAE loss as a single scalar tensor.
    """
    # Compute the weight mask based on the target
    weight_mask = compute_weight_mask(target)

    # Compute the absolute difference between the output and target
    absolute_diff = torch.abs(output - target)

    # Apply the weight mask to the absolute differences
    weighted_absolute_diff = torch.multiply(weight_mask, absolute_diff)

    # Sum up all the weighted absolute differences to get the final loss
    loss = torch.sum(weighted_absolute_diff)

    return loss

def CB_loss(output, target):
    """
    Compute a combined loss (CB_loss) that combines weighted MSE and weighted MAE.

    Parameters:
    - output: The predicted output tensor from the model.
    - target: The ground truth tensor.

    Returns:
    - combined_loss: A scalar tensor representing the combined loss.
    """
    # Compute the weighted MSE using the custom Bmse_loss function
    bmse = Bmse_loss(output, target)

    # Compute the weighted MAE using the custom Bmae_loss function
    bmae = Bmae_loss(output, target)

    # Combine the losses similar to the bmse_bmae_minmax function
    combined_loss = (bmse + 0.1 * bmae) / 2.0

    # Optionally, print the loss for debugging
    print("== CB_loss: ", combined_loss.item())

    return combined_loss

class Score:
    def __init__(self, thval, y_true, y_pred):
        self.thval = thval
        self.FO = torch.tensor(0.0)
        self.FX = torch.tensor(0.0)
        self.XO = torch.tensor(0.0)
        self.XX = torch.tensor(0.0)
        self.y_true = y_true
        self.y_pred = y_pred

        FO_0 = torch.sum(torch.where((self.y_pred >= self.thval) & (self.y_true >= self.thval), torch.tensor(1.0), torch.tensor(0.0)))
        self.FO = FO_0 + 1e-10  # To avoid division by zero
        self.FX = torch.sum(torch.where((self.y_pred >= self.thval) & (self.y_true < self.thval), torch.tensor(1.0), torch.tensor(0.0)))
        self.XO = torch.sum(torch.where((self.y_pred < self.thval) & (self.y_true >= self.thval), torch.tensor(1.0), torch.tensor(0.0)))
        self.XX = torch.sum(torch.where((self.y_pred < self.thval) & (self.y_true < self.thval), torch.tensor(1.0), torch.tensor(0.0)))

    def csi(self, weight):
        return weight * (1 - self.FO / (self.FO + self.FX + self.XO))

    def far(self, weight):
        return weight * (self.FX / (self.FO + self.FX))

def MCS_loss(y_pred,y_true):
    threshold = [0.1, 1, 2, 4, 6, 8, 10]
    weight = [20, 10, 5, 4, 3, 2, 1]  
    
    # Assuming conversion functions are applied here
    threshol = conversion(conversion(conversion(np.array(threshold), "Z_R", inverse=True), "DBZ_Z", inverse=True), "DN_DBZ", inverse=True)
    threshol = threshol / 160
    
    # Ensure the weight mask and calculations are on the same device as the y_pred tensor
    device = y_pred.device

    # Convert numpy array to a PyTorch tensor and move it to the same device as y_pred
    threshol = torch.tensor(threshol, dtype=torch.float32, device=device)
    weight = torch.tensor(weight, dtype=torch.float32, device=device)

    nthval = len(threshol)
    
    csi = 0
    far = 0
    for i in range(nthval):
        csi += Score(threshol[i], y_true, y_pred).csi(weight[i])
        far += Score(threshol[i], y_true, y_pred).far(weight[i])

    alpha = 0.00005
    bmse_bmae = CB_loss(y_true, y_pred)  
    mcs_loss = (bmse_bmae + alpha * (csi + far)) / 3.0
    print("== csi: ", alpha * csi.item())
    print("== far: ", alpha * far.item())
    print("== bmse_bmae: ", bmse_bmae.item())
    print("== mcs ", mcs_loss.item())
    
    return mcs_loss



