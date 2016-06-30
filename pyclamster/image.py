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
import logging
import os
import sys
import datetime
import copy

# External modules
import PIL.Image
import numpy as np
import scipy

# Internal modules
from . import coordinates
from . import fisheye


__version__ = "0.1"

# create logger
logger = logging.getLogger(__name__)


# image class
class Image(object):
    """
    class to deal with images. This class is basically a subclass to 
    PIL.Image.Image, but since it is not made for directly subclassing,
    this class is a wrapper that redirects attribute requests to an instance
    of class PIL.Image.Image.
    
    This class adds a simple possiblity to work with the underlying image
    data as numpy.ndarray. To get this array, use the Image.data property.
    You may also set this property to change the image. Note that currently
    only updating the image with information of the same image type is 
    possible.
    To get the underlying PIL.Image.Image, use the Image.image property. 
    You may also set this property to change the image.
    """

    ###################
    ### constructor ###
    ###################
    def __init__(self,
                 image=None,
                 time=None,
                 projection=None,
                 azimuth=None,
                 elevation=None,
                 zenith_pixel=None
                 ):  # TODO: fill with arguments
        """
        args:
            image(optional[PIL.Image.Image,str/path,Image]) something to read the image from
            time(optional[datetime.datetime]) time for image
            projection(optional[str]) projection used for image
            azimuth(optional[numpy array]): azimuth for each pixel - shape(width,height)
            elevation(optional[numpy array]): elevation for each pixel - shape(width,height)
            zenith_pixel(optional[numpy array]): pixel with 0 elevation and center point for azimuth - shape(2)
        """

        # set metadata
        self.time = time
        self.projection = projection
        self.coordinates = coordinates.Coordinates3d(
            azimuth = azimuth, elevation = elevation
            )
        self.zenith_pixel = zenith_pixel
        self.path = None

        # load the image
        self.loadImage(image)


    #############################################
    ### attributes/properties getters/setters ###
    #############################################
    # every attribute request (except _image itself) goes directly to _image
    # this makes this class practically a subclass to PIL.Image.Image
    def __getattr__(self, key):
        # logger.debug("requested attribute '{}'".format(key))
        if key == '_image':
            raise AttributeError(" ".join([
                "Can't access _image attribute.",
                "Did you try to access properties before",
                "loading an image?"
            ]))
        return getattr(self._image, key)

    @property
    def projection(self):
        return self._projection

    @projection.setter
    def projection(self, newprojection):
        self._projection = newprojection

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, newtime):
        if isinstance(newtime, datetime.datetime) or newtime is None:
            self._time = newtime
        else:
            raise ValueError("time has to be a datetime.datetime object.")

    @property
    def zenith_pixel(self):
        return self._zenith_pixel

    @zenith_pixel.setter
    def zenith_pixel(self, newzenith_pixel):
        self._zenith_pixel = newzenith_pixel

    # the image property is a wrapper around _image
    @property
    def image(self):
        return self._image

    # when you set the image property, both _image and _data are updated
    @image.setter
    def image(self, image):
        """
        set the underlying image and update the data

        args:
            newdata(PIL.Image.Image): the new image
        """
        if not isinstance(image, PIL.Image.Image):
            raise TypeError("image property has to be a PIL.Image.Image")

        # set values
        self._image = image
        self._data = np.array(self._image)
        # set coordinate shape
        self.coordinates.shape = self.data.shape[:2]


    @property
    def data(self):
        return self._data

    # when you set the data property, both _image and _data are updated
    @data.setter
    def data(self, newdata):
        """
        set the underlying image data and update the image. It is only possible
        to use the same image format (L,RGB,RGBA, etc...) as before.

        args:
            newdata(numpy.ndarray): the new image data, shape(width, height, {1,3})
        """
        try:  # check if image is set
            mode = self._image.mode
        except:  # no image set
            raise AttributeError(" ".join([
                "No image was specified until now.",
                "Can't determine image mode to set new data."
            ]))

        # set new data
        self._data = newdata
        self._image = PIL.Image.fromarray(self._data, mode)
        # set coordinate shape
        self.coordinates.shape = self.data.shape[:2]


    ###############
    ### methods ###
    ###############
    # try to read the time from image EXIF data
    def getEXIFtime(self, path=None):
        """
        get the EXIF time from either this image or an image specified by path

        args:
            path(optional[str/path]): an image to get the EXIF time from

        returns:
            datetime.datetime object or None
        """
        ret = None
        try:  # try to read time
            try:  # try to read Image from path
                image = PIL.Image.open(path)
            except:  # otherwise take current image
                image = self
            exif = image._getexif()  # read EXIF data
            t = exif[0x9003]  # get exif ctime value
            logger.info("EXIF ctime of image is '{}'".format(t))
            try:  # try to convert to datetime object
                t = datetime.datetime.strptime(str(t), "%Y:%m:%d %H:%M:%S")
                logger.debug(
                    "converted EXIF ctime to datetime object.")
                ret = t
            except:
                logger.warning(
                    "cannot convert EXIF ctime to datetime object.".format(t))
        except (AttributeError, ValueError, TypeError):  # reading didn't work
            logger.warning("cannot read EXIF time from image")

        return ret  # result

    # try to load the time from image EXIF data
    def loadEXIFtime(self, filename = None):
        if filename is None:
            filename = self.path
        if filename is None:
            logger.warning("No filename specified to read EXIF data.")
        logger.debug(
            "trying to load time from image '{}'".format(self.path))
        self.time = self.getEXIFtime( filename )

    # load the image
    def loadImage(self, image=None):
        """
        load image either from path, PIL.Image or numpy.ndarray
        
        args:
            image (str/path or PIL.Image or numpy.ndarray): image to load
        """
        ### create self._image according to specified argument ###
        success = False
        # looks like PIL image
        if isinstance(image, PIL.Image.Image):
            logger.info("reading image directly from PIL.Image.Image object")
            self.image = image
            success = True
        # argument is an image aleady
        elif isinstance(image, Image):
            logger.info("copying image directly from Image")
            # copy over attributes
            self.__dict__.update(image.__dict__)
            ### copy everything by hand ###
            self.time         = copy.deepcopy(image.time)
            self.projection   = copy.deepcopy(image.projection)
            self.coordinates  = copy.deepcopy(image.coordinates)
            self.zenith_pixel = copy.deepcopy(image.zenith_pixel)
            self.path         = copy.deepcopy(image.path)
            self.image        = copy.copy(image.image) # deepcopy not possible?

            success = True

        # argument looks like path
        elif isinstance(image, str):
            logger.debug("image argument is a string")
            if os.path.isfile(image):
                logger.debug("image argument is a valid path")
                logger.info("reading image from path")
                self.image = PIL.Image.open(image)
                self.path = image  # set path
                success = True
            else:
                logger.warning(
                    "image argument is not a valid path! Can't read image.")
                # self.image = PIL.Image.new(mode="RGB", size=(1, 1))
        # looks like numpy array
        elif isinstance(image, np.ndarray):
            logger.debug("argument is a numpy array")
            logger.debug("creating image from numpy array")
            self.path = None  # reset path because data comes from array
            raise Exception(" ".join([
                "Creating Image from ndarray is not implemented yet.",
                "use PIL.Image.fromarray and pass that to loadImage() instead."
            ]))
            self.data = image
            success = True
            # TODO: does not work like this, mode has to be specified somehow
        # nothing correct specified
        else:
            logger.info("nothing specified to read image. Nothing loaded.")
            # self.image = PIL.Image.new(mode="RGB", size=(1, 1)) # hard coded


    # load time from filename
    def _get_time_from_filename(self, fmt, filename=None):
        if isinstance(filename, str): f = filename
        else:                         f = self.path

        if not f is None:
            f = os.path.basename(f)
            return datetime.datetime.strptime(f, fmt)
        else:
            raise ValueError("Neither filename nor self.path is defined.")

    # try to load the time from filename
    def loadTimefromfilename(self, fmt, filename=None):
        self.time = self._get_time_from_filename(fmt, filename)

    # set time of image
    def setTime(self, time):
        """
        set internal image time
        
        args:
            time (datetime object): time to set
        """
        if isinstance(time, datetime.datetime):
            self.time = time
        else:
            logger.warning(
                "time is not a datetime object. Ignoring time setting.")

    def setMissingParameters(self):
        ### set zenith pixel ###
        if not isinstance(self.zenith_pixel, np.ndarray) \
                and isinstance(self.data, np.ndarray):
            self.zenith_pixel = self._calcCenter()

        ### set elevation-angle per pixel ###
        if not (isinstance(self.elevation_angle_per_pixel, float) or \
                        isinstance(self.elevation_angle_per_pixel, int)) and \
                isinstance(self.data, np.ndarray):
            self.elevation_angle_per_pixel = np.pi / self.data.shape[0]

    def _calcCenter(self):
        """
        calculates the center pixel of the image
        
        returns:
            center (numpy array): center pixel of the image - shape(2)
        """
        center = (np.array(self.data.shape) * 0.5).astype(int)
        return center

    ##########################
    ### Image manipulation ###
    ##########################
    def crop(self, box):
        """
        crop the image in-place to a box

        args:
            box (4-tuple of int): (left, top, right, bottom) (see PIL documentation)
        """

        # crop metadata
        # do this BEFORE re-setting the image
        # otherwise, the shapes wouldn't match and the coordinate class
        # would re-init the coordinates with empty masked arrays
        self.coordinates.crop(box)
        
        # crop image
        # somehow, self.image.crop( box ) alone does not work,
        # the self.image property has to be set...
        self.image = self.image.crop(box)

    def cut(self, box):
        """
        cut out a box of the image and return it

        args:
            box (4-tuple of int): (left, top, right, bottom) (see PIL documentation)

        returns:
            a new cut image
        """

        # copy image
        # deepcopy does not work somehow, so create a new image exactly like
        # this one
        cutimage = Image(self)

        # crop image
        cutimage.crop(box)

        return cutimage

    # ###################
    # Not needed anymore!
    # ###################
    # def cutNeighbour(self, center, nh_size=10, offset=(0, 0)):
    #     """
    #     Cut out a mini image around a center with given neighbourhood size.
    #     Args:
    #         center (tuple[int]): The position of the center (x, y).
    #         nh_size (int): The neighbourhood size.
    #         offset (optional[tuple[int]]): The possible offset of the center,
    #             due to cropped image (x,y).
    #
    #     Returns:
    #         mini_image (numpy.ndarray): The cutted mini image data.
    #     """
    #     box = [
    #         center[0] - nh_size + offset[0],
    #         center[1] - nh_size + offset[1],
    #         center[0] + nh_size + offset[0],
    #         center[1] + nh_size + offset[1]
    #         ]
    #     return self.cut(box).data

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

    def applyMask(self, mask):
        """
        args:
            mask (numpy mask): mask to be applied
        returns:
            maskedimage (Image): new instance of Image class with applied mask
        """
        pass

    ######################
    ### Transformation ###
    ######################
    def applyDistortionMap(self, map, inplace=False, order=0):
        # for some reason, isinstance(map, fisheye.DistortionMap)
        # or isinstance(map, DistortionMap) does not work?!
        # this solves it...
        if not map.__class__.__name__ == "DistortionMap":
            raise TypeError("map has to be a DistortionMap")

        if not np.shape(self.data)[:2] == map.src_shape:
            logger.warning("Map source shape is not defined or does not match!")

        if inplace: image = self # operate on this image
        else:       image = Image( self ) # copy over this image

        # apply map
        logger.debug("applying distortion map...")

        # This is NOT very elegant...
        # I don't know a better possibility to loop over the last index
        # than this. The problem is, that a grayscale image has no third index.
        # ...
        if len(np.shape(image.data)) == 3:
            for layer in range(np.shape(image.data)[2]):
                layerout = scipy.ndimage.interpolation.map_coordinates(
                    input = self.data[:,:,layer],
                    coordinates = map.T, # map has to be transposed somehow
                    order = order
                    )
                try:    out = np.dstack( (out, layerout) ) # add to stack
                except: out = layerout # first addition
            image.data = out # set image data
        else: # only 2 dim...
            image.data = scipy.ndimage.interpolation.map_coordinates(
                input = image.data,
                coordinates = map.T, # map has to be transposed somehow
                order = order
                )
            
        logger.debug("done applying distortion map.")

        if not inplace: # return new image if not inplace
            return image






if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = Image(data=np.zeros((50, 50)))
    plt.imshow(img.getAzimuth(), interpolation='none')
    plt.colorbar()
    plt.show()
