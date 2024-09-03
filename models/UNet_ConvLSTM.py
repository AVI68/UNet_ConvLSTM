#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:47:26 2024

@author: avijitmajhi
"""

# from typing import Tuple
# import tensorflow as tf
# import tensorflow.keras.layers as layers
# from tensorflow.keras import Model
# from tensorflow.keras.layers import (Activation, BatchNormalization, Conv3D,
#                           Conv3DTranspose, ConvLSTM2D, Dropout, Input, MaxPool3D)

# from tensorflow.keras.layers import Layer

# class ConvLSTMRecurrentLayer(Layer):
#     def __init__(self, num_filters_base, seq_len, **kwargs):
#         super(ConvLSTMRecurrentLayer, self).__init__(**kwargs)
#         self.num_filters_base = num_filters_base
#         self.seq_len = seq_len
#         self.convlstm_layer = ConvLSTM2D(filters=num_filters_base, kernel_size=(3, 3),
#                                           padding='same', return_sequences=True, return_state=True)
        
#     def call(self, tensor):
#         # Get the shape of the input tensor
#         batch_size = tf.shape(tensor)[0]
#         height = tf.shape(tensor)[2]
#         width = tf.shape(tensor)[3]
        
#         # Initialize the states
#         state_h = tf.zeros((batch_size, height, width, self.num_filters_base))
#         state_c = tf.zeros((batch_size, height, width, self.num_filters_base))
#         input_tensor = tf.zeros((batch_size, 1, height, width, self.num_filters_base))
        
#         # List to store the output at each timestep
#         out_fl = []
        
#         # Loop over the timesteps
#         for step in range(self.seq_len):
#             # Ensure that tensor has enough time steps
#             current_step = step % tf.shape(tensor)[1]
#             output_fl, state_h, state_c = self.convlstm_layer(
#                 input_tensor + tensor[:, current_step:current_step+1, :, :, :], 
#                 initial_state=[state_h, state_c]
#             )
#             out_fl.append(output_fl)  # No need to squeeze if the dimension isn't 1
#             input_tensor = output_fl
        
#         # Convert the list of outputs to a tensor and reshape it
#         out_fl = tf.stack(out_fl, axis=1)  # Shape: (batch_size, seq_len, height, width, num_filters_base)
        
#         return out_fl


# # Assuming ConvLSTMRecurrentLayer outputs a tensor of shape (batch_size, seq_len, height, width, channels)

# def unet_convlstm_reg(input_shape: Tuple[int] = (16, 256, 256, 1),
#                       num_filters_base: int = 16,
#                       dropout_rate: float = 0.2,
#                       seq_len: int = 15):  # Update seq_len to 15

#     inputs = layers.Input(shape=input_shape)
#     x_init = layers.BatchNormalization()(inputs)
#     x0 = x_init

#     x_conv1_b1 = ConvLSTM2D(filters=num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x0)
#     contex_b1 = layers.Cropping3D(cropping=((1,0),(0,0),(0,0)), data_format="channels_last")(x_conv1_b1)
#     out_fl_b1 = ConvLSTMRecurrentLayer(num_filters_base, seq_len)(contex_b1)
#     x_max_b1 = layers.MaxPool3D([1, 2, 2], padding='same')(x_conv1_b1)
#     x_bn_b1 = layers.BatchNormalization()(x_max_b1)
#     x_do_b1 = layers.Dropout(dropout_rate)(x_bn_b1)

#     x_conv1_b2 = ConvLSTM2D(filters=2*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_do_b1)
#     contex_b2 = layers.Cropping3D(cropping=((1,0),(0,0),(0,0)), data_format="channels_last")(x_conv1_b2)
#     out_fl_b2 = ConvLSTMRecurrentLayer(2*num_filters_base, seq_len)(contex_b2)
#     x_max_b2 = layers.MaxPool3D([1, 2, 2], padding='same')(x_conv1_b2)
#     x_bn_b2 = layers.BatchNormalization()(x_max_b2)
#     x_do_b2 = layers.Dropout(dropout_rate)(x_bn_b2)

#     x_conv1_b3 = ConvLSTM2D(filters=4*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_do_b2)
#     contex_b3 = layers.Cropping3D(cropping=((1,0),(0,0),(0,0)), data_format="channels_last")(x_conv1_b3)
#     out_fl_b3 = ConvLSTMRecurrentLayer(4*num_filters_base, seq_len)(contex_b3)
#     x_max_b3 = layers.MaxPool3D([1, 2, 2], padding='same')(x_conv1_b3)
#     x_bn_b3 = layers.BatchNormalization()(x_max_b3)
#     x_do_b3 = layers.Dropout(dropout_rate)(x_bn_b3)

#     x_conv1_b4 = ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_do_b3)
#     contex_b4 = layers.Cropping3D(cropping=((1,0),(0,0),(0,0)), data_format="channels_last")(x_conv1_b4)
#     out_fl_b4 = ConvLSTMRecurrentLayer(8*num_filters_base, seq_len)(contex_b4)
#     x_max_b4 = layers.MaxPool3D([1, 2, 2], padding='same')(x_conv1_b4)
#     x_bn_b4 = layers.BatchNormalization()(x_max_b4)
#     x_do_b4 = layers.Dropout(dropout_rate)(x_bn_b4)

#     x_conv1_b5 = ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same')(x_do_b4)
#     x_conv2_b5 = tf.expand_dims(x_conv1_b5, axis=1)
#     out_fl_b5 = ConvLSTMRecurrentLayer(8*num_filters_base, seq_len)(x_conv2_b5)

#     # Reshape out_fl_b5 to remove the unnecessary dimension before Conv3DTranspose
#     out_fl_b5_reshaped = tf.squeeze(out_fl_b5, axis=2)  # Remove the dimension of size 1

#     # Residual Decoder
#     x_deconv_b5 = layers.Conv3DTranspose(filters=8*num_filters_base, kernel_size=(1, 2, 2), strides=(1,2,2), padding='same', activation="relu")(out_fl_b5_reshaped)
#     x_bn_b5 = layers.BatchNormalization()(x_deconv_b5)
#     x_do_b5 = layers.Dropout(dropout_rate)(x_bn_b5)

#     # Ensure shapes are compatible before concatenation
#     out_fl_b4_reshaped = tf.squeeze(out_fl_b4, axis=2)  # Remove the dimension of size 1 from out_fl_b4

#     x_conv1_b6 = layers.Conv3D(filters=4*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(layers.concatenate([x_do_b5, out_fl_b4_reshaped]))
#     x_deconv_b6 = layers.Conv3DTranspose(filters=4*num_filters_base, kernel_size=(1, 2, 2), strides=(1,2,2), padding='same', activation="relu")(x_conv1_b6)
#     x_bn_b6 = layers.BatchNormalization()(x_deconv_b6)
#     x_do_b6 = layers.Dropout(dropout_rate)(x_bn_b6)

#     # Similarly, reshape other outputs before concatenation as needed
#     out_fl_b3_reshaped = tf.squeeze(out_fl_b3, axis=2)
#     x_conv1_b7 = layers.Conv3D(filters=2*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(layers.concatenate([x_do_b6, out_fl_b3_reshaped]))
#     x_deconv_b7 = layers.Conv3DTranspose(filters=2*num_filters_base, kernel_size=(1, 2, 2), strides=(1,2,2), padding='same', activation="relu")(x_conv1_b7)
#     x_bn_b7 = layers.BatchNormalization()(x_deconv_b7)
#     x_do_b7 = layers.Dropout(dropout_rate)(x_bn_b7)

#     out_fl_b2_reshaped = tf.squeeze(out_fl_b2, axis=2)
#     x_conv1_b8 = layers.Conv3D(filters=1*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(layers.concatenate([x_do_b7, out_fl_b2_reshaped]))
#     x_deconv_b8 = layers.Conv3DTranspose(filters=1*num_filters_base, kernel_size=(1, 2, 2), strides=(1,2,2), padding='same', activation="relu")(x_conv1_b8)
#     x_bn_b8 = layers.BatchNormalization()(x_deconv_b8)
#     x_do_b8 = layers.Dropout(dropout_rate)(x_bn_b8)

#     out_fl_b1_reshaped = tf.squeeze(out_fl_b1, axis=2)
#     x_conv1_b9 = layers.Conv3D(filters=1*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(layers.concatenate([out_fl_b1_reshaped, x_do_b8]))
#     residual_output = layers.Conv3D(1, kernel_size=(1, 1, 1), padding="same")(x_conv1_b9)
#     output = layers.Activation("linear", dtype="float32")(residual_output)

#     model = Model(inputs, output)
#     model.summary()

#     return model


### stepped down version


from typing import Tuple
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv3D,
                          Conv3DTranspose, ConvLSTM2D, Dropout, Input, MaxPool3D)

from tensorflow.keras.layers import Layer

class ConvLSTMRecurrentLayer(Layer):
    def __init__(self, num_filters_base, seq_len, **kwargs):
        super(ConvLSTMRecurrentLayer, self).__init__(**kwargs)
        self.num_filters_base = num_filters_base
        self.seq_len = seq_len
        self.convlstm_layer = ConvLSTM2D(filters=num_filters_base, kernel_size=(3, 3),
                                          padding='same', return_sequences=True, return_state=True)
        
    def call(self, tensor):
        # Get the shape of the input tensor
        batch_size = tf.shape(tensor)[0]
        height = tf.shape(tensor)[2]
        width = tf.shape(tensor)[3]
        
        # Initialize the states
        state_h = tf.zeros((batch_size, height, width, self.num_filters_base))
        state_c = tf.zeros((batch_size, height, width, self.num_filters_base))
        input_tensor = tf.zeros((batch_size, 1, height, width, self.num_filters_base))
        
        # List to store the output at each timestep
        out_fl = []
        
        # Loop over the timesteps
        for step in range(self.seq_len):
            # Ensure that tensor has enough time steps
            current_step = step % tf.shape(tensor)[1]
            output_fl, state_h, state_c = self.convlstm_layer(
                input_tensor + tensor[:, current_step:current_step+1, :, :, :], 
                initial_state=[state_h, state_c]
            )
            out_fl.append(output_fl)  # No need to squeeze if the dimension isn't 1
            input_tensor = output_fl
        
        # Convert the list of outputs to a tensor and reshape it
        out_fl = tf.stack(out_fl, axis=1)  # Shape: (batch_size, seq_len, height, width, num_filters_base)
        
        return out_fl


# Assuming ConvLSTMRecurrentLayer outputs a tensor of shape (batch_size, seq_len, height, width, channels)

def unet_convlstm_reg(input_shape: Tuple[int] = (4, 64, 64, 1),
                      num_filters_base: int = 16,
                      dropout_rate: float = 0.2,
                      seq_len: int = 3):  # Update seq_len to 15

    inputs = layers.Input(shape=input_shape)
    x_init = layers.BatchNormalization()(inputs)
    x0 = x_init

    x_conv1_b1 = ConvLSTM2D(filters=num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x0)
    contex_b1 = layers.Cropping3D(cropping=((1,0),(0,0),(0,0)), data_format="channels_last")(x_conv1_b1)
    out_fl_b1 = ConvLSTMRecurrentLayer(num_filters_base, seq_len)(contex_b1)
    x_max_b1 = layers.MaxPool3D([1, 2, 2], padding='same')(x_conv1_b1)
    x_bn_b1 = layers.BatchNormalization()(x_max_b1)
    x_do_b1 = layers.Dropout(dropout_rate)(x_bn_b1)

    x_conv1_b2 = ConvLSTM2D(filters=2*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_do_b1)
    contex_b2 = layers.Cropping3D(cropping=((1,0),(0,0),(0,0)), data_format="channels_last")(x_conv1_b2)
    out_fl_b2 = ConvLSTMRecurrentLayer(2*num_filters_base, seq_len)(contex_b2)
    x_max_b2 = layers.MaxPool3D([1, 2, 2], padding='same')(x_conv1_b2)
    x_bn_b2 = layers.BatchNormalization()(x_max_b2)
    x_do_b2 = layers.Dropout(dropout_rate)(x_bn_b2)

    x_conv1_b3 = ConvLSTM2D(filters=4*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_do_b2)
    contex_b3 = layers.Cropping3D(cropping=((1,0),(0,0),(0,0)), data_format="channels_last")(x_conv1_b3)
    out_fl_b3 = ConvLSTMRecurrentLayer(4*num_filters_base, seq_len)(contex_b3)
    x_max_b3 = layers.MaxPool3D([1, 2, 2], padding='same')(x_conv1_b3)
    x_bn_b3 = layers.BatchNormalization()(x_max_b3)
    x_do_b3 = layers.Dropout(dropout_rate)(x_bn_b3)

    x_conv1_b4 = ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same', return_sequences=True)(x_do_b3)
    contex_b4 = layers.Cropping3D(cropping=((1,0),(0,0),(0,0)), data_format="channels_last")(x_conv1_b4)
    out_fl_b4 = ConvLSTMRecurrentLayer(8*num_filters_base, seq_len)(contex_b4)
    x_max_b4 = layers.MaxPool3D([1, 2, 2], padding='same')(x_conv1_b4)
    x_bn_b4 = layers.BatchNormalization()(x_max_b4)
    x_do_b4 = layers.Dropout(dropout_rate)(x_bn_b4)

    x_conv1_b5 = ConvLSTM2D(filters=8*num_filters_base, kernel_size=(3,3), padding='same')(x_do_b4)
    x_conv2_b5 = tf.expand_dims(x_conv1_b5, axis=1)
    out_fl_b5 = ConvLSTMRecurrentLayer(8*num_filters_base, seq_len)(x_conv2_b5)

    # Reshape out_fl_b5 to remove the unnecessary dimension before Conv3DTranspose
    out_fl_b5_reshaped = tf.squeeze(out_fl_b5, axis=2)  # Remove the dimension of size 1

    # Residual Decoder
    x_deconv_b5 = layers.Conv3DTranspose(filters=8*num_filters_base, kernel_size=(1, 2, 2), strides=(1,2,2), padding='same', activation="relu")(out_fl_b5_reshaped)
    x_bn_b5 = layers.BatchNormalization()(x_deconv_b5)
    x_do_b5 = layers.Dropout(dropout_rate)(x_bn_b5)

    # Ensure shapes are compatible before concatenation
    out_fl_b4_reshaped = tf.squeeze(out_fl_b4, axis=2)  # Remove the dimension of size 1 from out_fl_b4

    x_conv1_b6 = layers.Conv3D(filters=4*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(layers.concatenate([x_do_b5, out_fl_b4_reshaped]))
    x_deconv_b6 = layers.Conv3DTranspose(filters=4*num_filters_base, kernel_size=(1, 2, 2), strides=(1,2,2), padding='same', activation="relu")(x_conv1_b6)
    x_bn_b6 = layers.BatchNormalization()(x_deconv_b6)
    x_do_b6 = layers.Dropout(dropout_rate)(x_bn_b6)

    # Similarly, reshape other outputs before concatenation as needed
    out_fl_b3_reshaped = tf.squeeze(out_fl_b3, axis=2)
    x_conv1_b7 = layers.Conv3D(filters=2*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(layers.concatenate([x_do_b6, out_fl_b3_reshaped]))
    x_deconv_b7 = layers.Conv3DTranspose(filters=2*num_filters_base, kernel_size=(1, 2, 2), strides=(1,2,2), padding='same', activation="relu")(x_conv1_b7)
    x_bn_b7 = layers.BatchNormalization()(x_deconv_b7)
    x_do_b7 = layers.Dropout(dropout_rate)(x_bn_b7)

    out_fl_b2_reshaped = tf.squeeze(out_fl_b2, axis=2)
    x_conv1_b8 = layers.Conv3D(filters=1*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(layers.concatenate([x_do_b7, out_fl_b2_reshaped]))
    x_deconv_b8 = layers.Conv3DTranspose(filters=1*num_filters_base, kernel_size=(1, 2, 2), strides=(1,2,2), padding='same', activation="relu")(x_conv1_b8)
    x_bn_b8 = layers.BatchNormalization()(x_deconv_b8)
    x_do_b8 = layers.Dropout(dropout_rate)(x_bn_b8)

    out_fl_b1_reshaped = tf.squeeze(out_fl_b1, axis=2)
    x_conv1_b9 = layers.Conv3D(filters=1*num_filters_base, kernel_size=(1,3,3), padding='same', activation="relu")(layers.concatenate([out_fl_b1_reshaped, x_do_b8]))
    residual_output = layers.Conv3D(1, kernel_size=(1, 1, 1), padding="same")(x_conv1_b9)
    output = layers.Activation("linear", dtype="float32")(residual_output)

    model = Model(inputs, output)
    model.summary()

    return model
