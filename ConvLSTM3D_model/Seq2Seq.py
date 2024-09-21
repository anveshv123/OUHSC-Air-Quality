import torch.nn as nn
import torch
import numpy as np
from .ConvLSTM3D import ConvLSTM3D


class BatchNorm4d(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm4d, self).__init__()
        
        self.batchnorm = nn.BatchNorm3d(num_features)
       

    def forward(self, X):
        shape=np.shape(X)
        return self.batchnorm(X.reshape(
            shape[0], shape[1], shape[2], shape[3], shape[4]*shape[5])).reshape(
                shape[0], shape[1], shape[2], shape[3], shape[4], shape[5])
        
      

class Seq2Seq(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size, num_layers):

        super(Seq2Seq, self).__init__()

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM3D(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)
        )

        self.sequential.add_module(
            "batchnorm1", BatchNorm4d(num_features=num_kernels)
        ) 

        # Add rest of the layers
        for l in range(2, num_layers+1):

            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM3D(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding, 
                    activation=activation, frame_size=frame_size)
                )
                
            self.sequential.add_module(
                f"batchnorm{l}", BatchNorm4d(num_features=num_kernels)
                ) 

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv3d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

    def forward(self, X):

        # Forward propagation through all the layers
        output = self.sequential(X)

        # Return only the last output frame
        output = self.conv(output[:,:,-1])
        
        return nn.Sigmoid()(output)

    