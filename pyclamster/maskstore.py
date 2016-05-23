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

# External modules

# Internal modules


__version__ = "0.1"


class MaskStore(object):
    def __init__(self):
        self.masks = {}

    def addMask(self, mask, label=None):
        """
        args:
            mask (numpy array): mask (0=pixel unused, 1=pixel used) - shape(width,height)
            label (optional[int or str]): label of the mask. Default automatic numbering
        """
        if label is None:
            pass

        self.masks[label] = mask

    def removeMask(self, label):
        """
        remove mask with label label from mask list
        args:
            label (int or str): label of the mask.
        """

        del self.masks[label]

    def getMask(self, labels=None):
        """
        return merged numpy array out of selected masks
        args:
            label (list of int or str): list of mask labels to be merged. Defaults to all masks.
        """
        pass
