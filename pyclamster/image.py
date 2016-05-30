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

# Internal modules


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
        self.azimuth = azimuth
        self.elevation = elevation
        self.zenith_pixel = zenith_pixel

        # load the image
        self.loadImage(image)

        # create empty elevation and azimuth
        try: # if image is already defined
            width, height, *rest  = np.shape(self.image)
            self.elevation = np.empty( (width, height) )
            self.azimuth   = np.empty( (width, height) )
        except: # image not yet defined
            # self.data      = np.empty( (0) ) # TODO does not work like this
            #self.elevation = np.empty( (0) )
            #self.azimuth   = np.empty( (0) )
            # don't do anything for now.
            pass
            

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


    #############################################
    ### attributes/properties getters/setters ###
    #############################################
    # every attribute request (except _image itself) goes directly to _image
    # this makes this class practically a subclass to PIL.Image.Image
    def __getattr__(self, key):
        #logger.debug("requested attribute '{}'".format(key))
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
        #if newprojection in (None,"equidistant","stereographic","orthogonal","equiangle"):
        self._projection = newprojection
        #else:
            #raise ValueError("unimplemented projection '{}'".format(newprojection))

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, newtime):
        if isinstance(newtime,datetime.datetime) or newtime is None:
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
        self._data  = np.array( self._image ) 

    # the data property is a wrapper around _data
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
        try: # check if image is set
            mode = self._image.mode
        except: # no image set
            raise AttributeError(" ".join([
                "No image was specified until now.",
                "Can't determine image mode to set new data."
                ]))

        # set new data
        self._data = newdata
        self._image = PIL.Image.fromarray( self._data , mode)


    @property
    def elevation(self):
        return self._elevation

    @elevation.setter
    def elevation(self, newelevation):
        if not newelevation is None:
            try:
                widthnew, heightnew  = np.shape(newelevation)
                #logger.debug("widthnew: {}, heightnew: {}".format(widthnew,heightnew))
                width, height, *rest = np.shape(self.image)
                #logger.debug("width: {}, height: {}".format(width,height))
                if (width,height) == (widthnew,heightnew):
                    self._elevation = newelevation
                else:
                    raise ValueError("shape of new elevation does not match image shape.")
            except:
                raise ValueError("elevation must be a numpy.ndarray with shape according to image.")
            
    @property
    def azimuth(self):
        return self._azimuth

    @azimuth.setter
    def azimuth(self, newazimuth):
        if not newazimuth is None:
            try:
                widthnew, heightnew  = np.shape(newazimuth)
                #logger.debug("widthnew: {}, heightnew: {}".format(widthnew,heightnew))
                width, height, *rest = np.shape(self.image)
                #logger.debug("width: {}, height: {}".format(width,height))
                if (width,height) == (widthnew,heightnew):
                    self._azimuth = newazimuth
                else:
                    raise ValueError("shape of new azimuth does not match image shape.")
            except:
                raise ValueError("azimuth must be a numpy.ndarray with shape according to image.")

    ###############
    ### methods ###
    ###############
    # try to load the time from image EXIF data
    def loadTime(self):
        try: # try to read time
            exif = self.image._getexif() # get EXIF data
            t = exif[0x9003] # get exif ctime value
            logger.info("EXIF ctime of image is '{}'".format(t))
            try: # try to convert to datetime object
                t = datetime.datetime.strptime(str(t), "%Y:%m:%d %H:%M:%S")
                self.setTime(t) # set time
                logger.debug(
                   "converted EXIF ctime to datetime object.")
            except:
                logger.warning(
                    "cannot convert EXIF ctime to datetime object.".format(t))
        except (AttributeError, ValueError, TypeError): # reading didn't work
            logger.warning("cannot read EXIF time from image")


    # load the image
    def loadImage(self, image=None):
        """
        load image either from path, PIL.Image or numpy.ndarray
        
        args:
            image (str/path or PIL.Image or numpy.ndarray): image to load
        """
        ### create self._image according to specified argument ###
        # looks like PIL image
        if isinstance(image, PIL.Image.Image):
            logger.info("reading image directly from PIL.Image.Image object")
            self.image = image
        # argument is an image aleady
        elif isinstance(image, Image):
            logger.info("copying image directly from Image")
            # copy over attributes
            self.__dict__.update(image.__dict__)

                
                
        # argument looks like path
        elif isinstance(image, str):
            logger.debug("image argument is a string")
            if os.path.isfile(image):
                logger.debug("image argument is a valid path")
                logger.info("reading image from path")
                self.image = PIL.Image.open(image)
            else:
                logger.warning(
                    "image argument is not a valid path! Can't read image.")
                #self.image = PIL.Image.new(mode="RGB", size=(1, 1))
        # looks like numpy array
        elif isinstance(image, np.ndarray):
            logger.debug("argument is a numpy array")
            logger.debug("creating image from numpy array")
            self.data = image
            # TODO: does not work like this, mode has to be specified somehow
        # nothing correct specified
        else:
            logger.info("nothing specified to read image. Nothing loaded.")
            #self.image = PIL.Image.new(mode="RGB", size=(1, 1)) # hard coded

        # init things
        #self.setDefaultParameters()
        #self.applyCameraCorrections()

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
            logger.warning("time is not a datetime object. Ignoring time setting.")

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
        center = np.round(np.array(self.data.shape) * 0.5)
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

        # crop image
        # somehow, self.image.crop( box ) alone does not work,
        # the self.image property has to be set...
        self.image = self.image.crop( box )
        # crop metadata
        self.elevation = self.elevation[box[1]:box[3],box[0]:box[2]] 
        self.azimuth   = self.azimuth  [box[1]:box[3],box[0]:box[2]]


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
        cutimage = Image( self )

        # crop image
        cutimage.crop( box )

        return cutimage

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
