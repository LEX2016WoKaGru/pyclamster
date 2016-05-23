# -*- coding: utf-8 -*-
"""
Created on 18.05.16
Created for pyClamster

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
import os,sys,time

# External modules
import scipy as sp
import numpy as np

# Internal modules


__version__ = "0.0.0.1"


class Camera(object):
    """
    class that holds a series of images
    and camera properties
    """
    def __init__(self):
        self.image_series     = {timestamp: Image}
        self.Azi_displacement = {timestamp: Azi}
        self.Ele_displacement = {timestamp: Ele}

    def getImage(self,timestamp):
        """
        return an instance of class Image() for the timestamp timestamp
        """
        image = Image(timestamp) # TODO
        return image


class Image(object):
    def __init__(self, path=None,data=None, elevation=None, azimuth=None):
        """
        args:
            data(numpy array): RBG values for each pixel  - shape(width,height,3)
            elevation(numpy array): elevation for each pixel - shape(width,height)
            azimuth(numpy array): azimuth for each pixel - shape(width,height)
        """
        self.data = data
        self.elevation = elevation
        self.azimuth = azimuth

        # load image from path if specified
        if isinstance(path,str) and os.path.isfile(path):
            self.loadImage(path)

    def loadImage(self, path):
        """
        read image from path
        """
        img = sp.ndimage.imread(path, mode="RGB")
        self.data = img

    def saveImage(self, path):
        """
        save image to path
        """
        img = sp.misc.imsave(path, self.data)

    def _calcCenter(self):
        pass

    def crop(self, px_x=960, px_y=960, center=True):
        center = self._calcCenter()
        self.data = self.data[center[0]-px_x:center[0]+px_x, center[1]]
        pass

    def cropRound(self, px=1920):
        pass

    def cropDegree(self, deg=45):
        pass

    def mean(self):
        pass

    def std(self):
        pass

    def getElevation(self,timestamp):
        """
        return elevation for each pixel as a numpy array
        
        """
        pass

    def getAzimuth(self):
        """
        return elevation for each pixel as a numpy array
        
        """
        pass

    def applyMask(self, mask):
        """
        args:
            mask (numpy mask): mask to be applied
        returns:
            maskedimage( Image ): new instance of Image class with applied mask
        """
        pass


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
        

class TrashDetector(object):
    def detect(self, image):
        return Image(image_wo_trash)


class Cloud(MaskedImage):
    def __add__(self, other):
        pass


class CloudMatcher(object):
    def match(self, Cloud1, Cloud2):
        pass

    def match_bulk(self, clouds=[]):
        pass

class WettermastData(object):
    pass


class Ceilometer(object):
    pass



if "__name__" == "__main__":
    raw = Image()
    raw.loadImage(path)
    raw.cropDegree(60)
    cleaned = TrashDetector().detect(raw)
    #=Cloud_cam1+Cloud_cam2
