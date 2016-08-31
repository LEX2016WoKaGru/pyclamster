#!/usr/bin/env python3
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
import os, glob, re
import datetime
import gc

# External modules
import logging
import PIL.Image

# Internal modules
from . import image


__version__ = "0.1"

# create logger
logger = logging.getLogger(__name__)

class CameraSession(object):
    """
    class that holds a series of images
    and camera properties
    """
    def __init__(self, images=None, times=None, fmt=None):
        """
        class constructor

        args:
            images (optional[list of filepaths or glob expression]): Specification of image files to use.
                Either a list of full filepaths or a glob expression. User directory is expanded. Defaults to None.
            times (optional[list of datetime objects]): Specification of time series corresponding to image files.
                Defaults to None, which means to try to extract the time from the embedded EXIF information.
        """
        # regex to check if filename seems to be an image file
        self.imagefile_regex = re.compile("\.(jpe?g)|(gif)|(png)", re.IGNORECASE)
        self.fmt = fmt

        # start with empty image and time series
        self.image_series = []
        self.time_series  = []

        # add images to internal series
        self.add_images( images, times, fmt=fmt )

        # Every camera session needs a calibration
        self.calibration = None

    def add_images(self, images, times=None, fmt=None):
        """
        add images to internal series

        args:
            images (list of filepaths or glob expression): Specification of image files to use.
                Either a list of full filepaths or a glob expression. User directory is expanded. Defaults to None.
            times (optional[list of datetime objects]): Specification of time series corresponding to image files.
                Defaults to None, which means to try to extract the time from the embedded EXIF information.
        """
        ### determine what kind the argument 'images' is ###
        # start with empty file list
        filelist = []

        ### argument is a list ###
        if isinstance(images, list):
            # filelist is directly the argument
            filelist = images

        ### argument is a string ###
        elif isinstance(images, str):
            ### looks like glob ###
            path = os.path.expanduser(images)
            gl = glob.glob( path ) # expand home directory and glob
            gl.sort()
            if gl: # the glob yielded someting
                logger.debug("'{}' looks like glob expression.".format(images))
                if len(gl) == 1 and os.path.isdir(path): # only one, may be a folder
                    logger.debug("'{}' is a folder. Search for image files in it...".format(path))
                    # filelist is all files in the folder
                    folder = gl[0] # folder ist single element
                    filelist = [os.path.join(folder,f) for f in os.listdir(folder)]
                else: # multiple globbed files
                    # filelist is all globbed files
                    logger.debug("glob expression '{}' yielded {} files".format(images,len(gl)))
                    filelist = gl

        # get images and time series from filelist
        images, time = self._get_images_and_time_series_from_filelist( filelist, times, fmt )

        # append found data to attribute
        self.image_series.extend( images )
        self.time_series.extend( time )

    def _get_images_and_time_series_from_filelist(self, files, times = None):
        """
        get image and time series from a list of full filepaths
        exclude files whose filename doesn't match the internal image regex self.imagefile_regex

        args:
            files (list of filepaths or glob expression): Specification of image files to use.
                Either a list of full filepaths or a glob expression. User directory is expanded. Defaults to None.
            times (optional[list of datetime objects]): Specification of time series corresponding to image files.
                Defaults to None, which means to try to extract the time from the embedded EXIF information.
            fmt (optional[str]): datetime.datetime.strptime format specification
                for determining the image time from its filename.
        returns:
            2-tuple of lists: (imageseries, timeseries)
        """

        # test if given images and times match
        if not times is None: # if times were specified
            if len(times) != len(files): # if lengths don't match
                logger.warning(
                    "non-matching images (len={}) and time (len={}) series specified. Dropping time series.".format(
                    len(files),len(times))
                    )
                times = [None for i in files] # list with only None
        else: # no time was specified
            logger.debug("no time was specified for any image.")
            times = [None for i in files] # list with only None

        # start with empty series
        image_series = []
        time_series = []

        # counters
        count_images = 0

        # find all image files from file list
        for f,t in zip(files,times): # iterate over all files
            if os.path.isfile(f): # is this an actual file?
                basename = os.path.basename(f) # basename
                if self.imagefile_regex.search(f): # if this looks like an image file
                    logger.debug("filename '{}' looks like an image file.".format(basename))
                    count_images = count_images + 1
                    if isinstance(t, datetime.datetime): # time was specified
                        logger.debug("A time was specified for image '{}'".format(basename))
                    elif t is None: # no time was specified
                        logger.debug("no time was specified directly for image '{}'.".format(basename))
                        # try to read time
                        # load the image
                        img = image.Image(f)
                        if isinstance(fmt,str): # try to get time from filename
                            logger.debug("searching in filename for time...")
                            t = img._get_time_from_filename(fmt=fmt,filename=f)
                        if t is None: # else try to get time from filename
                            logger.debug("searching in EXIF for time...")
                            t = img._get_time_from_filename(fmt=fmt,filename=f)
                            t = img.getEXIFtime()
                        del img; gc.collect() # free the memory for the image

                        if t is None:
                            logger.debug("time could not be determined automatically for image '{}'".format(basename))
                        else:
                            logger.debug("time for image '{}' is '{}'".format(basename,t))

                    else: # bogus was specified
                        logger.warning("the specified time for image '{}' is not a datetime object! Cannot set time.".format(basename))
                        
                        
    
                    # append data to series
                    image_series.append(f)
                    time_series.append(t)
                else:
                    logger.debug("filename '{}' does not look like an image file. skipping...".format(basename))


        count_times = sum(x is not None for x in time_series) # count available times
            
        logger.info("SUMMARY: {} files given, {} images found, {} times found".format(
            len(files),count_images,count_times)
            )
        if count_times != count_images: # number of images and times don't match
            logger.warning("{} image(s) have no associated time!".format(abs(count_times-count_images)))

        # return list of images and time
        return [image_series, time_series]


    def getImage(self,timestamp):
        """
        return an instance of class Image() for the timestamp timestamp
        """
        img = image.Image(timestamp) # TODO
        return img



###############
### Example ###
###############
if __name__ == "__main__":
    # logging setup
    logging.basicConfig(level = logging.INFO)

    # get list of images
    #images = os.path.expanduser("~/Bilder/*.jpg")
    images = "~/Bilder/2011-02-06-002.JPG"

    # create camera object
    c = Camera( images, times = [1] ) 
