# -*- coding: utf-8 -*-
"""
Created on 02.06.16

Created for pyclamster

    Copyright (C) {2016}

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in
    the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
# System modules
import random
from copy import deepcopy

# External modules
import scipy.ndimage
import numpy as np

# Internal modules
from ..image import Image

__version__ = "0.1"

"""
This module is for pre-processing transformations for the cloud detection
algorithms.
"""


def localBrightness(data, nh_size=10):
    """
    A function to standardize the data with the local brightness.
    Args:
        data (numpy array):
            The data field that should be transformed.
        nh_size (int): The neighbourhood size, in which the local brightness
            should be calculated.

    Returns:
        transformed_data (numpy array): The transformed data array.
    """
    len_dim2 = 1
    if len(data.shape) == 2:
        len_dim2 = data.shape[2]
    mean = scipy.ndimage.filters.uniform_filter(
        data, size=(nh_size, nh_size, len_dim2), mode="constant")
    return data - mean


def rbDetection(data):
    """
    A function to implement the red/blue cloud detection.
    Args:
        data (numpy array):
            The data field that should be transformed. The first entry of the
            second dimension should be the red array and
            the last entry the blue array.
    Returns:
        transformed_data (numpy array): The transformed data array.
    """
    blue = data[:, :, 0]
    red = data[:, :, -1]
    transformed_data = (blue - red) / (blue + red)
    transformed_data[blue + red == 0] = 1
    return transformed_data


def listShuffleSplit(lst, split_size=10):
    """
    Function to shuffle a list and to split up the list into parts with the same length.
    Args:
        lst (list): The input list.
        split_size (int): The splitted lists size.

    Returns:
        splitted_lists (list[list]): The splitted up lists with a length of size.
             The sublist could be smaller than size.
    """
    lst = deepcopy(lst)
    random.shuffle(lst)
    print(lst)
    splitted_lists = [
        lst[i:i + split_size] if (i + split_size) < len(lst) else lst[i:-1] for
        i in range(0, len(lst), split_size)]
    return splitted_lists
