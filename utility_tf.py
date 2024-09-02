#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 02:05:14 2024

@author: avijitmajhi
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2
import tensorflow as tf

def filelist(folder_path):
    """
    Lists all files in a given folder.
    """
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path) 
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
        mapped_image = np.clip(image, min_val, max_val)
        mapped_image = mapped_image.astype(np.float32)  # Convert to float for further processing

        filtered_image = cv2.medianBlur(mapped_image, ksize=3)
        resized_image = cv2.resize(filtered_image, (256, 256), interpolation=cv2.INTER_AREA)
        standardized_image = resized_image / max_val

        return standardized_image

    else:
        mapped_image = image * max_val
        mapped_image = np.clip(mapped_image, min_val, max_val)  # Clip values to stay within the 0-160 range
        mapped_image = mapped_image.astype(np.uint8)  # Convert to uint8 for image operations
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
        datetime_info = value.split(".")[0].split("\\")[-1]
        date_time_info = datetime.strptime(datetime_info, '%Y%m%d_%H%M')
        return date_time_info
    else:
        date_format = value.strftime('%Y%m%d_%H%M')
        filename = f"{date_format}.png"
        return filename

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
    R = np.where((R >= 0) & (R <= 120), R, 0)
    weights = weight(R)
    isw_score = np.sum(weights * R) 
    return isw_score


@tf.function
def compute_weight_mask(target):
    threshold = [0, 1, 5, 10, 20, 50]
    weights = [1., 2., 5., 10., 50.]

    threshol = conversion(conversion(conversion(np.array(threshold), "Z_R", inverse=True), "DBZ_Z", inverse=True), "DN_DBZ", inverse=True)
    threshol = threshol / 160

    threshol = tf.convert_to_tensor(threshol, dtype=tf.float32)
    mask = tf.ones_like(target, dtype=tf.float32)

    for k in range(len(weights)):
        mask = tf.where(
            (threshol[k] <= target) & (target < threshol[k + 1]), 
            tf.constant(weights[k], dtype=tf.float32), 
            mask
        )

    return mask

@tf.function
def Bmse_loss(output, target):
    weight_mask = compute_weight_mask(target)
    squared_diff = tf.square(output - target)
    weighted_squared_diff = tf.multiply(weight_mask, squared_diff)
    loss = tf.reduce_sum(weighted_squared_diff)
    return loss

@tf.function
def Bmae_loss(output, target):
    weight_mask = compute_weight_mask(target)
    absolute_diff = tf.abs(output - target)
    weighted_absolute_diff = tf.multiply(weight_mask, absolute_diff)
    loss = tf.reduce_sum(weighted_absolute_diff)
    return loss

@tf.function
def CB_loss(output, target):
    bmse = Bmse_loss(output, target)
    bmae = Bmae_loss(output, target)
    combined_loss = (bmse + 0.1 * bmae) / 2.0

    tf.print("== CB_loss: ", combined_loss)

    return combined_loss

@tf.function
def MCS_loss(y_pred, y_true):
    threshold = [0.1, 1, 2, 4, 6, 8, 10]
    weight = [20, 10, 5, 4, 3, 2, 1]  
    
    threshol = conversion(conversion(conversion(np.array(threshold), "Z_R", inverse=True), "DBZ_Z", inverse=True), "DN_DBZ", inverse=True)
    threshol = threshol / 160
    
    threshol = tf.convert_to_tensor(threshol, dtype=tf.float32)
    weight = tf.convert_to_tensor(weight, dtype=tf.float32)

    nthval = len(threshol)
    
    csi = tf.constant(0.0)
    far = tf.constant(0.0)
    for i in tf.range(nthval):
        score_obj = Score(threshol[i], y_true, y_pred)
        csi += score_obj.csi(weight[i])
        far += score_obj.far(weight[i])

    alpha = 0.00005
    bmse_bmae = CB_loss(y_true, y_pred)  
    mcs_loss = (bmse_bmae + alpha * (csi + far)) / 3.0

    tf.print("== csi: ", alpha * csi)
    tf.print("== far: ", alpha * far)
    tf.print("== bmse_bmae: ", bmse_bmae)
    tf.print("== mcs ", mcs_loss)
    
    return mcs_loss
class Score:
    def __init__(self, thval, y_true, y_pred):
        self.thval = thval
        self.FO = tf.constant(0.0)
        self.FX = tf.constant(0.0)
        self.XO = tf.constant(0.0)
        self.XX = tf.constant(0.0)
        self.y_true = y_true
        self.y_pred = y_pred

        FO_0 = tf.reduce_sum(tf.where((self.y_pred >= self.thval) & (self.y_true >= self.thval), 1.0, 0.0))
        self.FO = FO_0 + 1e-10  # To avoid division by zero
        self.FX = tf.reduce_sum(tf.where((self.y_pred >= self.thval) & (self.y_true < self.thval), 1.0, 0.0))
        self.XO = tf.reduce_sum(tf.where((self.y_pred < self.thval) & (self.y_true >= self.thval), 1.0, 0.0))
        self.XX = tf.reduce_sum(tf.where((self.y_pred < self.thval) & (self.y_true < self.thval), 1.0, 0.0))

    @tf.function
    def csi(self, weight):
        return weight * (1 - self.FO / (self.FO + self.FX + self.XO))

    @tf.function
    def far(self, weight):
        return weight * (self.FX / (self.FO + self.FX))
