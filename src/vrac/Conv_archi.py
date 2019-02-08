import numpy as np
import pandas as pd
from math import floor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import itertools as it

from tqdm import tqdm


def conv_block(in_filter, output_filter, nb_conv, kernel_size=3,
               activation_function=nn.ReLU()):
    """To simplify the creation of convolutional sequences

    Parameters
    ----------
    in_filter :  int
        Number of filters that we want in entry

    output_filter :  int
        Number of filters that we want in output

    nb_conv : int
        Number of convolution layers

    activation_function : nn Function
        Activation function after each convolution

    Returns
    ---------
    sequential : Sequential torch Object
        The convolutional sequence that we were seeking
    """
    nbchannel = in_filter
    nbfilter = output_filter
    sequential = []

    for i in range(nb_conv):
        sequential.append(
            nn.Conv2d(nbchannel, nbfilter, kernel_size, padding=1))
        sequential.append(activation_function)
        nbchannel = nbfilter
    sequential.append(nn.MaxPool2d(2))
    return sequential


def network_from_shape(net_structure, activation=nn.ReLU()):
    """To simplify the creation of fully connected layers sequences

    Parameters
    ----------
    net structure: int list
        Describe each layer size -> one entry of the list is a layer conv_size

    activation_function : nn Function
        Activation function after each layer of the net

    Returns
    ---------
    temp :  Torch object list
        The fully connected sequence with the last activation function "tanh"
    """
    temp = []
    for prev, next in zip(net_structure[:-1], net_structure[1:]):
        temp.append(nn.Linear(prev, next))
        temp.append(activation)
    temp = temp[:-1]  # Remove last activation return temp
    return temp


class Convolutional(nn.Module):
    """A class for convolutional architectures"""

    def __init__(self, conv_sizes, img_shape, network_lin_shape, speed_size):
        """Creating network

        Parameters
        ----------
        conv_sizes :  int tuple list
            A list containing a tuple for each convolutionnal
            layer -> (number of filters, number of convolutions layers,
                      kernel size)

        img_shape : int couple
            A tuple describing image's height and with

        network_lin_shape : int tuple
            A tuple describing the size of each fully connected layer

        speed_size : int
            Size of speed one-hot encoding
        """
        super(nn.Module, self).__init__()

        conv_sizes = [(img_shape[0],)] + conv_sizes
        layers = it.chain(*[conv_block(prev[0], curr[0], curr[1], curr[2])
                            for prev, curr in zip(conv_sizes, conv_sizes[1:])])
        self.conv = nn.Sequential(*layers)
        # print(self.conv)
        img_flatten_size = img_shape[1:]
        # Use of formula in :$
        # https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d
        # to calculate input size for first layer of the fully connected block
        for i in range(len(conv_sizes) - 1):
            img_flatten_size = floor((img_flatten_size[0] + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1), floor(
                (img_flatten_size[1] + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)
        img_flatten_size = img_flatten_size[0] * \
            img_flatten_size[1] * conv_sizes[-1][0]
        self.fc = network_from_shape(
            (img_flatten_size + speed_size,) + network_lin_shape + (2,),
            activation=nn.ReLU())
        self.fc = nn.Sequential(*self.fc)
        # print(self.fc)

    def forward(self, img, speed):
        """Passing img and speed input as describe in NVIDIA architecture

        Parameters
        ----------
        img :  int tuple
            (Height, width, channels)

        speed : float Tensor
            One hot vector -> One hot because if we concatenate just one
            value to the flattened output ouf convolutional sequence, speed
            will be not really taken in account for training
        """
        output = self.conv(img)
        output = torch.cat((output.reshape(output.size(0), -1), speed), dim=1)
        output = self.fc(output)
        return output


if __name__ == '__main__':
    pass
