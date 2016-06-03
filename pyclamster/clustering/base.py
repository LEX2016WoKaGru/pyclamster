# -*- coding: utf-8 -*-
"""
Created on 03.06.16

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
import abc

# External modules

# Internal modules


__version__ = "0.1"


class Scaler(object):
    """
    Base class for the pre-processing steps to cluster the data
    """
    def __init__(self):
        self.parameters = {}
        self.fitted = False

    @abc.abstractmethod
    def fit(self, data):
        """
        Method to train the object, the train parameters were set.
        Args:
            data (numpy array): The input data for training of the instance.
        """
        pass

    @abc.abstractmethod
    def transform(self, data):
        """
        Method to transform the data with fitted parameters.
        Args:
            data (numpy array): The data which should the transformed.

        Returns:
            transformed_data (numpy array/float): The transformed data.
                The return type depends on the transformation.
        """
        pass

    def fit_transform(self, data):
        """
        Method to train the object and to transform the given data in the same
        step. Calls the methods fit and transform to do this step.

        Args:
            data (numpy array): The training data which should the transformed.

        Returns:
            transformed_data (numpy array/float): The transformed data.
                The return type depends on the transformation.
        """
        self.fit(data)
        return self.transform(data)


