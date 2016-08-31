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
    def __init__(self, images=None):
        """
        class constructor

        args:
            images (optional[list of filepaths or glob expression]): Specification of image files to use.
                Either a list of full filepaths or a glob expression. User directory is expanded. Defaults to None.
        """
        # regex to check if filename seems to be an image file
        self.imagefile_regex = re.compile("\.(jpe?g)|(gif)|(png)", re.IGNORECASE)

        # start with empty image and time series
        self.image_series = []

        # add images to internal series
        self.add_images( images )

        # Every camera session needs a calibration
        self.calibration = None

    def add_images(self, images ):
        """
        add images to internal series

        args:
            images (list of filepaths or glob expression): Specification of image files to use.
                Either a list of full filepaths or a glob expression. User directory is expanded. Defaults to None.
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
        images = self._get_images_from_filelist( filelist )

        # append found data to attribute
        self.image_series.extend( images )

    def _get_images_from_filelist(self, files):
        """
        get image and time series from a list of full filepaths
        exclude files whose filename doesn't match the internal image regex self.imagefile_regex

        args:
            files (list of filepaths or glob expression): Specification of image files to use.
                Either a list of full filepaths or a glob expression. User directory is expanded. Defaults to None.
        returns:
            list of valid image paths
        """

        # start with empty series
        image_series = []

        # counters
        count_images = 0

        # find all image files from file list
        for f in files: # iterate over all files
            if os.path.isfile(f): # is this an actual file?
                basename = os.path.basename(f) # basename
                if self.imagefile_regex.search(f): # if this looks like an image file
                    logger.debug("filename '{}' looks like an image file.".format(basename))
                    count_images = count_images + 1
                    # append data to series
                    image_series.append(f)
                else:
                    logger.debug("filename '{}' does not look like an image file. skipping...".format(basename))


        logger.info("SUMMARY: {} files given, {} images found".format(
            len(files),count_images)
            )
        # return list of images and time
        return image_series


    ###################################################################
    ### make the CameraSession behave correct in certain situations ###
    ###################################################################
#    # make it iterable
#    def __iter__(self):
#        try: del self.current # reset counter
#        except: pass
#        return self
#
#    def __next__(self):
#        try:    self.current += 1 # try to count up
#        except: self.current  = 0 # if that didn't work, start with 0
#        if self.current >= len(self.methods):
#            del self.current # reset counter
#            raise StopIteration # stop the iteration
#        else:
#            return self.image_series[self.current]
        
    # make it indexable
    def __getitem__(self,key):
        return image.Image(self.image_series[key])


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
