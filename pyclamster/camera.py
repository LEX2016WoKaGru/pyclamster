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
import logging
import pickle

# External modules
import numpy as np

# Internal modules
from . import image
from . import utils
from . import calibration
from . import fisheye
from . import coordinates
from . import positioning


__version__ = "0.1"

# create logger
logger = logging.getLogger(__name__)

class CameraSession(object):
    """
    class that holds a series of images
    and camera properties
    """
    def __init__(self, longitude, latitude, heightNN, imgshape, smallshape, 
        rectshape, zone, calibration=None, images=None, distmap=None):
        """
        class constructor

        args:
            images (optional[list of filepaths or glob expression]): 
                Specification of image files to use.  Either a list of full 
                filepaths or a glob expression. User directory is expanded. 
                Defaults to None.
            latitude,longitude (float): gps position of image in degrees
            heightNN (float): height in metres over NN
            imgshape (tuple of int): shape of images
            smallshape (tuple of int): shape of smaller resized images
            rectshape (tuple of int): shape of rectified images
            zone (int): zone for UTM
            calibration (optional[pyclamster.calibration.CameraCalibration]): 
                Calibration of camera session
            distmap (optional[pyclamster.fisheye.DistortionMap]): Distortion 
                map
        """
        # regex to check if filename seems to be an image file
        self.imagefile_regex = re.compile("\.(jpe?g)|(gif)|(png)", re.IGNORECASE)

        # start with empty image and time series
        self.reset_images()

        # add images to internal series
        self.add_images( images )

        # image meta data
        self.imgshape = imgshape
        self.smallshape = smallshape
        self.rectshape = rectshape

        # Every camera session needs a calibration
        self.calibration = calibration

        # camera session position
        self.longitude = longitude
        self.latitude  = latitude
        self.heightNN  = heightNN

        # calculate position
        self.position = self.calculate_carthesian_coordinates(zone=zone)
        self.position.z = heightNN


    def reset_images(self):
        """
        empty the list of used files
        """
        self.image_series = []

    def set_images(self, images ):
        """
        use this as images
        """
        self.reset_images()
        self.add_images(images)

    def add_images(self, images ):
        """
        add images to internal series

        args:
            images (list of filepaths or glob expression): Specification of 
                image files to use.  Either a list of full filepaths or a glob 
                expression. User directory is expanded. Defaults to None.
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
        exclude files whose filename doesn't match the internal image regex 
        self.imagefile_regex

        args:
            files (list of filepaths or glob expression): Specification of 
            image files to use.  Either a list of full filepaths or a glob 
            expression. User directory is expanded. Defaults to None.
        returns:
            list of valid image paths
        """

        # start with empty series
        image_series = []
        files.sort()

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

    ### projection to carthesian coordinates
    def calculate_carthesian_coordinates(self,zone):
        proj = positioning.Projection(zone=zone) 
        return proj.lonlat2xy(self.longitude, self.latitude
            ,return_coordinates = True)


    ###################################################################
    ### make the CameraSession behave correct in certain situations ###
    ###################################################################
    # make it indexable
    def __getitem__(self,key):
        img = image.Image(
            image = self.image_series[key],
            longitude = self.longitude,
            latitude  = self.latitude
            )
        # resize image to smaller size
        img.image = img.resize(self.smallshape)
        img.image = img.resize(self.smallshape)
        # rectify
        img.coordinates = self.calibration.create_coordinates(
            (img.data.shape[:2]))
        img.position = self.position  # set carthesian position
        img.applyDistortionMap(self.distmap, inplace=True)
        return img

    ##############################
    ### DistortionMap creation ###
    ##############################
    def createDistortionMap(self, rectshape=None, max_angle=utils.deg2rad(45)):
        """
        create a distortion map for rectification, set it as property and
        return it as well.
        
        args:
            rectshape (optional[tuple of int]): shape of rectified image
            max_angle (float): maximum angle (radians) from optical axis to be
                rectified (measured to SIDE of image, not corner)
        returns:
            distmap = pyclamster.fisheye.DistortionMap
        """
        if rectshape is None:
            if self.rectshape is None:
                raise TypeError("rect_shape has to be defined somehow.")
            else:
                rect_shape = self.rectshape

        ### Create rectified coordinates ###
        rect_x,rect_y=np.meshgrid(
            np.linspace(-1,1,num=rect_shape[1]),# image x coordinate goes right
            np.linspace(1,-1,num=rect_shape[0]) # image y coordinate goes up
            )
        # set virtual height of layer to get azimuth and elevation
        rect_z = 1 / np.tan(max_angle)
        
        rect_coord = coordinates.Coordinates3d(
            x = rect_x, y = rect_y, z = rect_z,
            azimuth_offset = 3/2 * np.pi,
            azimuth_clockwise = True,
            shape=rect_shape
            )

        ### create image coordinates ###
        if isinstance(self.calibration, calibration.CameraCalibration):
            img_coord = self.calibration.create_coordinates(self.smallshape)
        else:
            raise Exception("No calibration given for CameraSession.")

        ### calculate recfication map ###
        logger.debug("calculating rectification map...")
        distmap = fisheye.FisheyeProjection.distortionMap(
            in_coord=img_coord, out_coord=rect_coord, method="nearest"
            ,basedon="spherical")
        logger.debug("done calculating rectification map!")

        self.distmap = distmap
        return distmap

    def iterate_over_images(self,size=None):
        """
        yield one image after another
        """
        if not size is None:
            coords = self.calibration.create_coordinates(size)
        else:
            coords = self.calibration.create_coordinates(self.imgshape)
        for imgpath in self.image_series:
            # load image
            img = image.Image(imgpath)
            if not size is None:
                img.image = img.resize(size)
            # set coordinates
            img.coordinates = coords
            img.position = self.position # set carthesian position
            logger.debug("yielding image '{}'".format(imgpath))
            yield img
        

    # iterate_over_rectified_images
    def iterate_over_rectified_images(self):
        """
        yield one rectified image after another
        """
        for imgpath in self.image_series:
            # load image
            img = image.Image(imgpath)
            # resize image to smaller size
            img.image = img.resize(self.smallshape)
            # rectify
            img.coordinates = self.calibration.create_coordinates(
                (img.data.shape[:2]))
            img.position = self.position # set carthesian position
            img.applyDistortionMap(self.distmap, inplace=True)
            logger.debug("yielding rectified image '{}'".format(imgpath))
            yield img
        
    # save this object to file
    def save(self, file):
        fh = open(file,"wb")
        logging.debug("pickling myself to file '{}'".format(file))
        pickle.dump(self, fh)
        fh.close()
            
