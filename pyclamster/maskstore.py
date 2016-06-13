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

# External modules
import numpy as np
import scipy.ndimage

# Internal modules
from .image import Image as pyClImage

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

    def applyMasks(self, image, labels=None, replace=True):
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
            image = pyClImage(image)
        if image.data.shape[2]>1:
            w, h = mask.shape
            mask = np.reshape(mask, (w, h, 1))
            mask = np.dstack((mask,mask,mask))
        image.data = np.ma.masked_array(image, mask=mask)
        return image


def denoiseMask(mask, denoising_ratio=5):
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
    eroded_mask = scipy.ndimage.binary_erosion(
        mask, structure=np.ones((denoising_ratio, denoising_ratio)))
    denoised_mask = scipy.ndimage.binary_propagation(
        eroded_mask, structure=np.ones((denoising_ratio, denoising_ratio)),
        mask=mask)
    # opened_mask = scipy.ndimage.binary_opening(
    #     mask, structure=np.ones((denoising_ratio, denoising_ratio)))
    # denoised_mask = scipy.ndimage.binary_opening(
    #     opened_mask, structure=np.ones((denoising_ratio, denoising_ratio)))
    denoised_mask = ~denoised_mask
    return denoised_mask
