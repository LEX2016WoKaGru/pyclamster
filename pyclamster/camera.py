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

# External modules
import logging
from PIL import Image

# Internal modules
import image


__version__ = "0.1"

# create logger
logger = logging.getLogger(__name__)

class Camera(object):
    """
    class that holds a series of images
    and camera properties
    """
    def __init__(self, images=None):
        # regex to check if filename seems to be an image file
        self.imagefile_regex = re.compile("\.(jpe?g)|(gif)|(png)", re.IGNORECASE)
        # start with empty image and time series
        self.image_series = []
        self.time_series  = []

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
            gl = glob.glob(images)
            if gl: # the glob yielded someting
                logger.debug("'{}' looks like glob expression.".format(images))
                if len(gl) == 1 and os.path.isdir(images): # only one, may be a folder
                    logger.debug("'{}' is a folder. Search for image files in it...".format(images))
                    # filelist is all files in the folder
                    folder = gl[0] # folder ist single element
                    filelist = [os.path.join(folder,f) for f in os.listdir(folder)]
                else: # multiple globbed files
                    # filelist is all globbed files
                    logger.debug("'{}' yielded {} files".format(images,len(gl)))
                    filelist = gl

        # get images and time series from filelist
        images, time = self._get_images_and_time_series_from_filelist( filelist )

        # append found data to attribute
        self.image_series.extend( images )
        self.time_series.extend( time )

        #self.Azi_displacement = {timestamp: Azi}
        #self.Ele_displacement = {timestamp: Ele}

    def _get_images_and_time_series_from_filelist(self, files):
        # start with empty series
        image_series = []
        time_series = []

        # counters
        count_times  = 0
        count_images = 0

        # find all image files from file list
        for f in files: # iterate over all files
            basename = os.path.basename(f) # basename
            if self.imagefile_regex.search(f): # if this looks like an image file
                time = None # start with empty time
                logger.debug("filename '{}' looks like an image file.".format(basename))
                img  = Image.open(f) # open image
                count_images += 1 # count up images found
                if hasattr(img, '_getexif'): # check if EXIF data is available
                    logger.debug("reading EXIF data...")
                    exif = img._getexif() # get EXIF data
                    try: # try to read time
                        time = exif[0x9003] # get exif ctime value
                        count_times += 1 # count up times
                    except: # reading time didn't work
                        logger.warning("cannot read EXIF time from image '{}'.".format(basename))
                else: # no EXIF data available
                    logger.warning("cannot read EXIF time from image '{}'.".format(basename))
                    time = None

                # append data to series
                image_series.append(f)
                time_series.append(time)
            else:
                logger.debug("filename '{}' does not look like an image file. skipping...".format(basename))

        logger.debug("SUMMARY: {} files given, {} images found, {} images with times found".format(len(files),count_images,count_times))

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
    logging.basicConfig(level = logging.DEBUG)

    # get list of images
    images = os.path.expanduser("~/Bilder")

    # create camera object
    c = Camera( images ) 
