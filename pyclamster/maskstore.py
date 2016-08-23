# -*- coding: utf-8 -*-
"""
Created on 23.05.16

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
import operator
import functools
from copy import deepcopy

# External modules
import numpy as np
import scipy.ndimage

from skimage.morphology import remove_small_objects

# Internal modules
from .image import Image as pyClImage
from .matching.cloud import Cloud

__version__ = "0.1"


class MaskStore(object):
    """
    This class is a container for numpy masks.
    """
    def __init__(self, masks={}):
        """
        Args:
            masks (optional[dict/list[numpy array]]): A dictionary or list of
                numpy array masks. A list will be automatically transformed
                into a dictionary also the elements are transformed into a
                boolean array. The default is an empty dictionary.
        """
        self.masks = masks

    @property
    def masks(self):
        return self._masks

    @masks.setter
    def masks(self, masks):
        if isinstance(masks, dict):
            masks_value = list(masks.values())
            masks_keys = list(masks.keys())
        elif hasattr(masks, '__iter__'):
            masks_value = list(masks)
            masks_keys = range(0, len(masks_value))
        else:
            raise ValueError(
                "The input masks have to be within a dict or an "
                "iterable object")
        # Tranformation into a boolean array
        masks_value = [m.astype(bool) for m in masks_value]
        self._masks = dict(zip(masks_keys, masks_value))

    def addMask(self, mask, label=None):
        """
        args:
            mask (numpy array): This is the mask which should be appended to
                the store (0/False=pixel unused, 1/True=pixel used). The mask
                will be automatically transformed into a boolean array.
                The shape is width*height.
            label (optional[int or str]): label of the mask.
                Default is None, which means automatic numbering.
        """
        if label is None:
            pass
        assert mask is np.ndarray, 'The mask have to be a numpy array.'
        mask = mask.astype(bool)
        self.masks[label] = mask

    def removeMask(self, label):
        """
        remove mask with label label from mask list
        args:
            label (int or str): label of the mask which should be removed.
        """
        del self.masks[label]

    def getMask(self, labels=None):
        """
        return merged numpy boolean array out of selected masks.
        args:
            label (optional[list of int or str]): list of mask labels to be
                merged. Defaults to all masks.
        returns:
            merged_masks (numpy array): A single mask in which the selected
             masks are merged.
        """
        if labels is None :
            masks = self.masks.values()
        else:
            masks = [self.masks[key] for key in labels]
        if len(labels)>1:
            merged_masks = functools.reduce(np.ma.mask_or, masks, 1).astype(bool)
        else:
            merged_masks = masks[0]
        return merged_masks

    def denoise(self, labels=None, denoising_ratio=5):
        """
        Denoise selected masks.
        Args:
            label (optional[list of int or str]): list of mask labels to be
                denoised. Defaults to all masks.
            denoising_ratio (int): The ratio within which pixels the
                denoising step will be executed.

        Returns:
            self:
        """
        if labels is None :
            keys = self.masks.keys()
        else:
            keys = [key for key in labels]
        for key in keys:
            self.masks[key] = denoiseMask(self.masks[key], denoising_ratio)
        return self

    def labelMask(self, labels=None):
        """
        Label a given mask.
        Args:
            labels (optional[list of int or str]): list of mask labels to be
                labeled. Defaults to all masks.

        Returns:
            labels (Labels): Labels instance with the labels as data.
            n_labels (int): Number of labels.
        """
        mask = self.getMask(labels)
        labels, nb_labels = scipy.ndimage.label(~mask)
        return Labels(labels), nb_labels

    def getCloud(self, image, labels=None):
        """
        Method to get a cloud instance with a given image for selected labels.
        Args:
            image (Image): Image instance of the base image for the cloud.
            labels (optional[list of int or str]): list of mask labels to be
                selected. Defaults to all masks.

        Returns:
            cloud_image (Cloud): Cloud instance, with only the mask based on
                the selected labels.
        """
        data = self.applyMask(image, labels)
        return Cloud(data)


    def applyMask(self, image, labels=None, fill_value=None, replace=True):
        """
        Mask an pyClImage instance with selected masks.
        Args:
            image (pyClImage): The pyClImage instance which should be masked.
            labels (optional[list[str/int]]):  list of mask labels for the
                Image mask. Defaults to all masks.
            replace (optional[bool]): If the given Image instance should be
                replaced by the masked image.

        Returns:
            masked_image (pyClImage): The pyClImage instance with masked data.
        """
        mask = self.getMask(labels)
        if replace:
            image = pyClImage(image=image)
        if image.data.shape[2]>1:
            w, h = mask.shape
            mask = np.reshape(mask, (w, h, 1))
            mask = np.dstack((mask,mask,mask))
        image.data = np.ma.masked_array(image, mask=mask)
        if not fill_value is None:
            image.data.set_fill_value(fill_value)
        return image

    def cutMask(self, image, labels=None):
        mask = (~self.getMask(labels)).astype(int)
        nozero = np.nonzero(mask[:,:])
        cut = [np.min(nozero[1]), np.min(nozero[0]), np.max(nozero[1]), np.max(nozero[0])]
        return image.cut(cut)
    # TODO Method to cut a mask out of an image


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
                converted in True. The other labels are set as False.
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
            return self
        else:
            return Labels(reshaped_labels)

    def getMaskStore(self):
        """
        Method to convert the labels into a mask store.
        Returns:
            converted_store (MaskStore): The converted MaskStore with the
                unique labels as masks.
        """
        converted_store = {}
        for val in np.unique(self.labels):
            converted_store[val] = (self.labels!=val)
        return MaskStore(converted_store)


def denoiseMask(mask, denoising_ratio=15):
    """
    Function to denoise a mask represented by a numpy array. The denoising is
    done with binary erosion and propagation.
    Args:
        mask (numpy array): The mask which should be denoised represented by a
            boolean numpy array.
        denoising_ratio (int): The ratio within which pixels the denoising step
            will be executed.
    Returns:
        denoised_mask (numpy array): The denoised mask represented by a boolean
            numpy array.
    """
    mask = ~mask
    # eroded_mask = scipy.ndimage.binary_erosion(
    #     mask, structure=np.ones((denoising_ratio, denoising_ratio)))
    # denoised_mask = scipy.ndimage.binary_propagation(
    #     eroded_mask, structure=np.ones((denoising_ratio, denoising_ratio)),
    #     mask=mask)
    # opened_mask = scipy.ndimage.binary_opening(
    #     mask, structure=np.ones((denoising_ratio, denoising_ratio)))
    # denoised_mask = scipy.ndimage.binary_opening(
    #     opened_mask, structure=np.ones((denoising_ratio, denoising_ratio)))
    denoised_mask = remove_small_objects(mask, denoising_ratio)
    denoised_mask = ~denoised_mask
    return denoised_mask
