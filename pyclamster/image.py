# -*- coding: utf-8 -*-
"""
Created on 23.05.16

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

# External modules
import scipy.ndimage
import scipy.misc

# Internal modules


__version__ = "0.1"


class Image(object):
    def __init__(self, path=None, data=None, elevation=None, azimuth=None):
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
        if isinstance(path, str) and os.path.isfile(path):
            self.loadImage(path)

    def loadImage(self, path):
        """
        read image from path
        """
        img = scipy.ndimage.imread(path, mode="RGB")
        self.data = img

    def saveImage(self, path):
        """
        save image to path
        """
        img = scipy.misc.imsave(path, self.data)

    def _calcCenter(self):
        pass

    def crop(self, px_x=960, px_y=960, center=True):
        center = self._calcCenter()
        self.data = self.data[center[0] - px_x:center[0] + px_x, center[1]]
        pass

    def cropRound(self, px=1920):
        pass

    def cropDegree(self, deg=45):
        pass

    def mean(self):
        pass

    def std(self):
        pass

    def getElevation(self, timestamp):
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