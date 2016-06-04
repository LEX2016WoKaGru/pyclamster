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

# External modules
import scipy
import numpy as np

# Internal modules
from .base import Scaler

__version__ = "0.1"

"""
This module is for pre-processing transformations for the cloud detection
algorithms.
"""


class LocalBrightness(Scaler):
    def __init__(self, neighbourhood_size=10, mode="constant", replace=False):
        assert isinstance(neighbourhood_size, int)
        self.nb_size = neighbourhood_size
        self.mode = mode
        self.parameters = None
        self.fitted = False
        self.replace = False

    def fit(self, data):
        len_dim2 = 1
        if len(data.shape) == 2:
            len_dim2 = data.shape[2]
        self.parameters = scipy.ndimage.filters.uniform_filter(
            data, size=(self.nb_size, self.nb_size, len_dim2), mode=self.mode)
        self.fitted = True
        return self

    def transform(self, data):
        if self.fitted:
            return data - self.parameters


class RBDetection(Scaler):
    """
    Class to implement the RBDetection algorithm.
    """

    def __init__(self, n_red=0, n_blue=2):
        """
        Args:
            n_red (optional[int]): The index of the red array. Default is 0.
            n_blue (optional[int]): The index of the blue array. Default is 2.
        """
        self.params = {"n_red": n_red, "n_blue": n_blue}

    def fit(self, data):
        """
        No setting of parameters is needed, this method is only for consistency
        purpose.
        Args:
            data (numpy array): Input data
        """
        return

    def transform(self, data):
        blue = data[:, :, self.params["n_blue"]]
        red = data[:, :, self.params["n_red"]]
        transformed_data = (blue-red)/(blue+red)
        transformed_data[blue+red==0] = 1
        return transformed_data
