# -*- coding: utf-8 -*-
"""
Created on 05.06.16

Created for pyclamster

@author: Tobias Sebastian Finn, tobias.sebastian.finn@studium.uni-hamburg.de

    Copyright (C) {2016}  {Tobias Sebastian Finn}

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
from copy import deepcopy

# External modules
import numpy as np
import scipy
import scipy.ndimage
import scipy.misc
from sklearn.feature_extraction.image import extract_patches_2d

# Internal modules


__version__ = "0.1"


class Labels(object):
    """
    This class contains the labels of the K-Means algorithm.
    The labels could be splitted up, reshaped, filtered.
    """

    def __init__(self, labels):
        """
        The initialization of the labels class.
        Args:
            labels (numpy array): The array, with the labels.
        """
        self.labels = labels

    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, labels):
        self.__labels = labels

    def binarize(self, labels_true, replace=False):
        """
        Method to convert the labels into boolean values.
        Args:
            labels_true (int/list[int]): The label/labels which should be
                converted in True. The other labels aare set as False.
            replace (bool): If the original labels should be replaced.

        Returns:
            bin_labels (optional[Labels]): If replace is False,
                the binarized labels will be returned as new Labels instance.
        """
        if not replace:
            bin_labels = deepcopy(self.labels)
        else:
            bin_labels = self.labels
        bin_labels[bin_labels in labels_true] = True
        bin_labels[not bin_labels in labels_true] = False
        if not replace:
            return Labels(bin_labels)

    def splitUp(self, **kwargs):
        """
        Method to split the labels up. This a wrapper for the numpy.split
        function. For more informations and arguments please refer to the
        documentation of numpy.split (the array has not to be specified
        in the arguments).
        Returns:
            splitted_labels (list[Labels]):
                The splitted up labels as list with new Labels instances.
        """
        splitted_labels = np.split(self.labels, **kwargs)
        splitted_labels = [Labels(l) for l in splitted_labels]
        return splitted_labels

    def reshape(self, new_shape, replace=False, **kwargs):
        """
        Method to reshape a labels class with given size. This is a wrapper for
        the numpy.reshape function. For more informations please
        refer to the documentation of numpy.reshape.
        Args:
            new_shape (int/tuple[int]): The new shape should have the same
                elements sum as the old shape.
            replace (bool): If the original labels should be replaced.

        Returns:
            reshaped_labels (optional[Labels]): If replace is False,
                the reshaped labels will be returned as new Labels instance.
        """
        reshaped_labels = np.reshape(self.labels, newshape=new_shape)
        if replace:
            self.labels = reshaped_labels
        else:
            return Labels(reshaped_labels)

    # def getRelevantSamples(self, nh_size=12, crit=0.75):
    #     xy_size = nh_size*2+1
    #     patches = extract_patches_2d(self.labels,(xy_size, xy_size))
    #     y_samples = []
    #     x_samples = []
    #     for i in range(patches.shape[0]):
    #         val = patches[i, nh_size+1, nh_size+1]
    #         unique, counts = np.unique(patches[i,:,:], return_counts=True)
    #         counts = dict(zip(unique, counts))
    #         if counts[val]>crit*xy_size*xy_size:
    #             y_samples.append(val)
    #             x_samples.append(i)
    #     return x_samples, y_samples


    # def filterRelevants(self, nh_size=5, replace=False):
    #     """
    #     Method to constrain the labels only to labels,
    #     where in a given neighbourhood the labels are the  same.
    #     Args:
    #         nh_size (Optional[int]): The size of the neighbourhood.
    #             Default is 5.
    #         replace (bool): If the original labels should be replaced.
    #
    #     Returns:
    #         filtered_labels (optional[Labels]): If replace is False,
    #             the filtered labels will be returned as new Labels instance.
    #     """
    #     mask = None
    #     for val in [0, 1]:
    #         print(val)
    #         temp = deepcopy(self.labels)
    #         temp[temp==val] = True
    #         temp[temp!=val] = False
    #         temp = temp.astype(int)
    #         temp = scipy.ndimage.binary_erosion(temp, structure=np.ones((nh_size, nh_size)))
    #         temp = temp.astype(bool)
    #         print(self.labels[temp])
    #         if mask is None:
    #             mask = temp
    #         else:
    #             mask = mask | temp
    #     if replace:
    #         self.mask = mask
    #     else:
    #         return self.labels[mask]
    #
    # def getLabelSamples(self):
    #     """
    #     Method to get the positions and labels out of a Labels instance.
    #     Returns:
    #         positions (list[tuple[int]]): The label positions.
    #         labels (list[int/bool]): The labels as integer or tuple
    #             as the original labels.
    #     """
    #     labels = self.labels.ravel().tolist()
    #     if not self.mask is None:
    #         positions = np.asarray(np.where(self.mask)).T.tolist()
    #     else:
    #         positions = np.asarray(np.where(self.labels>-999999)).T.tolist()
    #     return positions, labels


