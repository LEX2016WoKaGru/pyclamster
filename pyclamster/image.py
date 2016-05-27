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
import logging
import os
import sys

# External modules
import scipy.ndimage
import PIL.Image
import scipy.misc
import numpy as np

# Internal modules


__version__ = "0.1"

logger = logging.getLogger(__name__)


class Image(object):
    def __init__(self, image=None, path=None,
                 data=None):  # TODO: fill with arguments
        """
        args:
            image(optional[PIL.Image.Image]) image as PIL.Image.Image
            path(optional[str]): path to import image data form
            data(optional[numpy array]): RBG values for each pixel  - shape(width,height,3)
            azimuth(optional[numpy array]): azimuth for each pixel - shape(width,height)
            elevation(optional[numpy array]): elevation for each pixel - shape(width,height)
            zenith_pixel(optional[numpy array]): pixel with 0 elevation and center point for azimuth - shape(2)
            elevation_angle_per_pixel(optional[float]): elevation angle per pixel (assuming equidistant projection)
            azimuth_north_angle(optional[float]): offset from mathematical north 
        """
        if isinstance(image, PIL.Image.Image):
            logger.debug("reading image directly from PIL.Image.Image object")
            self._image = image
        elif isinstance(path, str):
            logger.debug("path argument is specified and a string")
            if os.path.isfile(path):
                logger.debug("path argument is a valid path")
                self._image = PIL.Image.open(path)
            else:
                logger.warning(
                    "path argument is not a valid path! Can't read image.")
                self._image = PIL.Image.new(mode="RGB", size=(1, 1))
        elif isinstance(data, np.ndarray):
            logger.debug("data argument is specified and a numpy array")
            self._image = PIL.Image.fromarray(newdata,
                                              "RGB")  # TODO hard-coded RGB here...
        else:
            logger.warning("nothing specified to read image.")
            self._image = PIL.Image.new(mode="RGB", size=(1, 1))

        #        self.azimuth = azimuth
        #        self.elevation = elevation
        #        self.zenith_pixel = zenith_pixel #TODO add possibility to adjust zenith_pixel for camera correction
        #        self.elevation_angle_per_pixel = elevation_angle_per_pixel
        #        self.azimuth_north_angle = azimuth_north_angle
        #        self.azimuth_direction = azimuth_direction
        #        self.timestamp = timestamp

        # load image from path if specified
        #        if isinstance(path, str) and os.path.isfile(path):
        #            self.loadImage(path)
        #        else:
        #            self.setMissingParameters()

    # every attribute request (except _image itself goes directly to self._image)
    # this makes this class practically a subclass to PIL.Image.Image
    def __getattr__(self, key):
        if key == '_image':
            #  http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
            raise AttributeError()
        return getattr(self._image, key)


    # if you request data attribute, this method is called
    @property
    def data(self):
        return np.array(self._image)

    # if you set self.data = newdata, this method is called
    @data.setter
    def data(self, newdata):
        self._image = PIL.Image.fromarray(newdata, self._image.mode)

    def setMissingParameters(self):
        ### set zenith pixel ###
        if not isinstance(self.zenith_pixel, np.ndarray) and isinstance(
                self.data, np.ndarray):
            self.zenith_pixel = self._calcCenter()

        ### set elevation-angle per pixel ###
        if not (isinstance(self.elevation_angle_per_pixel, float) or \
                        isinstance(self.elevation_angle_per_pixel, int)) and \
                isinstance(self.data, np.ndarray):
            self.elevation_angle_per_pixel = np.pi / self.data.shape[0]

    def loadImage(self, path):
        """
        read image from given path and store the RGB-values in self.data
        
        args:
            path (str): path to load the image from - shape(1)
        """
        img = scipy.ndimage.imread(path, mode="RGB")
        self.data = img
        self.setDefaultParameters()
        self.applyCameraCorrections()

    def saveImage(self, path):
        """
        save image RGB-values to given path
        
        args:
            path (str): path to store the image to - shape(1)
        """
        img = scipy.misc.imsave(path, self.data)

    def _calcCenter(self):
        """
        calculates the center pixel of the image
        
        returns:
            center (numpy array): center pixel of the image - shape(2)
        """
        center = np.round(np.array(self.data.shape) * 0.5)
        return center

    def crop(self, px_x=960, px_y=960, center=None):
        """
        cut rectangle out of the image around the pixel specified by 'center'
        there for use crop margins 
        
        returns:
            cropped_image (Image): image with rectangle cut out of data - data.shape(px_x * 2, px_y * 2)
        """
        # check if given center is in bounds or set center to center of image if not given
        if isinstance(center, np.ndarray):
            if center[0] < 0 or center[0] > self.data.shape[0] or \
                            center[1] < 0 or center[1] > self.data.shape[1]:
                logger.error(
                    "can't crop image, center pixel out of bounds (center = (%d, %d))" %
                    center[0], center[1])
                sys.exit()
        else:
            center = self._calcCenter()

        # calculate margins for new image
        xbounds = (center[0] - px_x, center[0] + px_x)
        ybounds = (center[1] - px_y, center[1] + px_y)

        # test if new margins are within the current image
        if xbounds[0] < 0 or \
                        xbounds[1] > self.data.shape[0] or \
                        xbounds[0] > xbounds[1]:
            logger.error(
                "can't crop image, requested margins out of bounds (xbounds = (%d, %d))" % xbounds)
            logger.debug("image margins = (%d, %d))" % (0, self.data.shape[0]))
            sys.exit()

        if ybounds[0] < 0 or \
                        ybounds[1] > self.data.shape[1] or \
                        ybounds[0] > ybounds[1]:
            logger.error(
                "can't crop image, requested margins out of bounds (ybounds = (%d, %d))" % ybounds)
            logger.debug("image margins = (%d, %d))" % (0, self.data.shape[1]))
            sys.exit()

        logger.debug("xbounds %d %d" % xbounds)
        logger.debug("ybounds %d %d" % ybounds)

        ### crop image and reset corresponding values ###
        # crop data
        cropped_data = self.data[xbounds[0]: xbounds[1],
                       ybounds[0]: ybounds[1]]

        # crop zenith
        cropped_zenith = self.zenith_pixel - np.array([xbounds[0], ybounds[0]])
        logger.debug(
            "old zenith %d %d" % (self.zenith_pixel[0], self.zenith_pixel[1]))
        logger.debug(
            "new zenith %d %d" % (cropped_zenith[0], cropped_zenith[1]))

        # crop elevation if existent
        cropped_elevation = None
        if isinstance(self.elevation, np.ndarray):
            cropped_elevation = self.elevation[xbounds[0]: xbounds[1],
                                ybounds[0]: ybounds[1]]

        # crop azimuth if existent
        cropped_azimuth = None
        if isinstance(self.azimuth, np.ndarray):
            cropped_azimuth = self.azimuth[xbounds[0]: xbounds[1],
                              ybounds[0]: ybounds[1]]

        ### create new image ###
        cropped_image = Image(data=cropped_data,
                              azimuth=cropped_azimuth,
                              elevation=cropped_elevation,
                              zenith_pixel=cropped_zenith,
                              elevation_angle_per_pixel=self.elevation_angle_per_pixel,
                              azimuth_north_angle=self.azimuth_north_angle,
                              azimuth_direction=self.azimuth_direction,
                              timestamp=self.timestamp)
        return cropped_image

    def cropDegree(self, deg=45. * np.pi / 180.):
        """
        cut a circle around the 'center' pixel of the image
        
        args:
            deg (optional[float]): max degrees of view the image should be cut to (max = 90.0 degree)
        returns:
            cropped_image (Image): image with circle cut out of data - data.shape(px_x * 2, px_y * 2)
        """
        # TODO check if given values are in bounds
        # TODO if deg used calc px
        # TODO check if px > self.data.shape * 0.5:  cut not possible
        # TODO set border-values to NaN -> mask np.array
        #     x, y = np.mgrid[:cropped_data.shape[0],:cropped_data.shape[1]]
        #     r    = deg / 90.0 * self.data.shape[0] # if deg used
        #  OR r    = px                              # if px  used 
        #     mask = x**2 + y**2 < r**2
        # TODO crop elevation
        # TODO crop azimuth
        # TODO add Image init variables
        cropped_image = Image()
        return cropped_image

    def getElevation(self):
        # TODO only equidistant -> create _getEquidistantElevation()
        """
        store the elevation for each pixel as a numpy array
        """
        # TODO return
        x, y = np.mgrid[:self.data.shape[0], :self.data.shape[1]]
        x = x - self.zenith_pixel[0]
        y = y - self.zenith_pixel[1]
        r = np.sqrt(x ** 2 + y ** 2)
        elevation = r * self.elevation_angle_per_pixel
        return elevation

    def getAzimuth(self):
        """
        store the azimuth for each pixel as a numpy array
        """
        x, y = np.mgrid[:self.data.shape[0], :self.data.shape[1]]
        x = x - self.zenith_pixel[0]
        y = y - self.zenith_pixel[1]

        if self.azimuth_direction != 'anticlockwise':
            x = -x

        azimuth = np.arctan2(x, -y) + np.pi + self.azimuth_north_angle
        azimuth[azimuth > 2 * np.pi] = azimuth[azimuth > 2 * np.pi] - 2 * np.pi
        return azimuth

    def applyMask(self, mask):
        """
        args:
            mask (numpy mask): mask to be applied
        returns:
            maskedimage (Image): new instance of Image class with applied mask
        """
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = Image(data=np.zeros((50, 50)))
    plt.imshow(img.getAzimuth(), interpolation='none')
    plt.colorbar()
    plt.show()
