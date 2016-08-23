# -*- coding: utf-8 -*-
"""
Created on 10.06.16

Created for pyclamster

    Copyright (C) {2016}

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
# System modules

# External modules
import numpy as np

import scipy.ndimage
import scipy.linalg

from sklearn.base import BaseEstimator, TransformerMixin


# Internal modules


class LCN(BaseEstimator, TransformerMixin):
    """
    LCN is a class to normalize an image with the so called local contrast
    normalization.
    """
    def __init__(self, size=None, patches=False, estimator=None,
                 copy=True, mean=True, scale=False):
        if estimator is None:
            self.base_estimator_ = scipy.ndimage.uniform_filter
        else:
            self.base_estimator_ = estimator
        if patches:
            self.mode_ = 'reflect'
            self.size = size
        else:
            self.mode_ = 'constant'
            self.size = size
        self.copy = copy
        self.mean_ = None
        self.scale_ = None
        self.mean = mean
        self.scale = scale

    def fit(self, X):
        """
        Method to calculate the means and the scales.
        Args:
            X (numpy array): The used data to calculate the mean and the
                standard deviation along the features axis for scaling purpose.
                Shape should be (n_height, n_width, n_channels).

        Returns:
            self:
        """
        if self.size is None:
            self.size = X.shape
        try:
            self.mean_ = self.base_estimator_(X, size=self.size, mode=self.mode_)
            squared = self.base_estimator_(X**2, size=self.size, mode=self.mode_)
        except:
            self.mean_ = self.base_estimator_(X, mode=self.mode_)
            squared = self.base_estimator_(X**2, mode=self.mode_)
        var_ = squared - self.mean_**2
        self.scale_ = np.sqrt(var_)
        self.scale_[self.scale_<np.mean(self.scale_)] = np.mean(self.scale_)
        return self

    def transform(self, X):
        """
        Method to perform a saved local contrast normalization to an image.

        Args:
            X (numpy array): The image which should be scaled.
                Shape should be (n_height, n_width, n_channels).

        Returns:
            X (numpy array): The transformed image.
                The shape is the same as the input array.
        """
        if self.copy:
            X = np.array(X, copy=self.copy)
            X = np.copy(X)
        if self.mean:
            X -= self.mean_
        if self.scale:
            X = X/self.scale_
        return X


class ZCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, bias=.1, copy=True):
        self.n_components = n_components
        self.bias = bias
        self.copy = copy

        self.input_shape_ = None
        self.mean_ = None
        self.components_ = None

    def fit(self, X, y=None):
        X = np.array(X, copy=self.copy)
        self.input_shape_ = X.shape
        X = X.reshape((X.shape[0], -1))
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        U, S, VT = scipy.linalg.svd(X, full_matrices=False)
        components = np.dot(VT.T * np.sqrt(1.0 / (S ** 2 + self.bias)), VT)
        self.components_ = components[:self.n_components]
        return self

    def transform(self, X):
        if self.copy:
            X = np.array(X, copy=self.copy)
        input_shape = X.shape
        if len(input_shape)>1:
            X = X.reshape((input_shape[0], -1))
        else:
            X = X.reshape((1, -1))
        X -= self.mean_
        X = np.dot(X, self.components_.T)
        X = X.reshape(input_shape)
        return X
