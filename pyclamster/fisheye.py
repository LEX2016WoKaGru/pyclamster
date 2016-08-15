# -*- coding: utf-8 -*-
"""
Created on 30.05.16

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

# External modules
import numpy as np
import numpy.ma as ma
import scipy.interpolate

# Internal modules
from . import coordinates


__version__ = "0.1"

# create logger
logger = logging.getLogger(__name__)


# fisheye class
class FisheyeProjection(object):
    """
    class for utils to handle fisheye projections
    """
    def __init__(self, projection=None):
        # currently implemented fisheye projections
        self._implemented_projections = ("equidistant")

        self.projection = projection


    ##################
    ### Properties ###
    ##################
    @property
    def projection(self):   
        return self._projection

    @projection.setter
    def projection(self, newprojection):
        if not newprojection is None:
            if newprojection in self._implemented_projections:
                self._projection = newprojection
            else: # if not implemented, raise error
                raise ValueError("unimplemented fisheye projection '{}'".format(
                    newprojection))
        else:
            self._projection = None

    ###############
    ### Methods ###
    ###############
    def createFisheyeElevation(self, 
                               shape,
                               center=None,
                               maxele=None,
                               maxelepos=None,
                               projection=None
                               ):
        """
        create a np.ndarray that holds an elevation for each pixel of an image
        according to fisheye projection type

        args:
            shape(2-tuple of int): shape of image (height, width)
            center(optional[2-tuple of int]): center/optical axis position on image (row,col).
                if not specified, the center of the image is assumed.
            maxelepos(optional[2-tuple of int]): position of pixel with maxele visible (row,col).
                if not specified, the left center of the image is assumed.
            maxele(optional[float]): the maximum angle (rad) from optical axis/center visible on the image

        returns:
            np.maskedarray of shape (width, height) with elevation values
        """
        # shape
        try:
            height, width = shape
        except:
            raise ValueError("shape argument not defined as 2-tuple.")

        # center
        try:
            center_row, center_col = center
        except:
            logger.debug("center not specified as 2-tuple, assuming image center")
            center_row, center_col = int(height/2),int(width/2) # image center

        # maxele
        if maxele is None:
            logger.debug("maxele not defined, assuming 90 degrees.")
            maxele = np.pi / 2

        # maxele pixel position
        try:
            maxele_row, maxele_col = maxelepos
        except:
            logger.debug("maxele position not specified as 2-tuple, assuming left image center")
            maxele_row, maxele_col = int(height/2),int(0) # left center


        # projection
        if projection is None: # projection defined?
            projection = self.projection # default projection from object
        if projection is None: # if projection is still not defined
            raise ValueError("no fisheye projection specified.")

            
        # width, height:          width and height of resulting array
        # center_row, center_col: center position / optical axis
        # maxele:               maximum angle visible on resulting array
        # projection:             fisheye projection type

        # now create arrays based on projection
        if projection == "equidistant":
            ### equidistant projection ###
            # create grid with only rows and cols
            row, col = np.mgrid[:height, :width]
            # center rows and cols
            col = col - center_col
            row = row - center_row
            # calculate radius from center
            r = np.sqrt(col ** 2 + row ** 2)
            # maximum radius from center / norming value
            norm = np.sqrt( (center_row - maxele_row) ** 2 + \
                            (center_col - maxele_col) ** 2 )

            # norm the radius to the maximum angle
            elevation = r / norm * maxele 

            
        else:
            raise ValueError(" ".join((
                "unimplemented fisheye projection '{}'",
                "You should never see this...")
                ).format(projection))

        # mask everything outside the sensible region
        elevation = ma.masked_where(                                   \
            # condition: everything outside the maxele region
            r > norm,        \
            # array
            elevation                                                  \
            )

        # return
        return elevation


    def createAzimuth(self,
                      shape,
                      center=None, 
                      maxelepos=None, 
                      clockwise=False,
                      north_angle=np.pi/2,
                      ):
        """
        create a np.ndarray that holds an azimuth for each pixel of an image

        args:
            shape(2-tuple of int): shape of image (height, width)
            center(optional[2-tuple of int]): center/optical axis position on image (row,col).
                if not specified, the center of the image is assumed.
            maxelepos(optional[2-tuple of int]): position of pixel with max elevation angle visible (row,col).
                if not specified, the left center of the image is assumed.
            clockwise(optional[boolean]): does the azimuth go clockwise? 
                Defaults to False (mathematical direction)
            north_angle(optional[float]): north angle on the image (rad). 
                Imagine a standard carthesian coordinate system lying on the 
                image. The north_angle is the angle between the positive x-axis
                and the 0-azimuth line depending on clockwise argument. 
                Defaults to 90°, pi/2, which is the top of the image at 
                counter-clockwise direction.

        returns:
            np.maskedarray of shape (width, height) with azimuth values
        """
        # shape
        try:
            height, width = shape
        except:
            raise ValueError("shape argument not defined as 2-tuple.")

        # center
        try:
            center_row, center_col = center
        except:
            logger.debug("center not specified as 2-tuple, assuming image center")
            center_row, center_col = int(height/2),int(width/2) # image center
            
        # maxele pixel position
        try:
            maxele_row, maxele_col = (None, None)
            maxele_row, maxele_col = maxelepos
        except:
            logger.debug("maxele position not specified as 2-tuple, assuming no border.") 
            #maxele_row, maxele_col = int(height/2),int(0) # left center

        # width, height:          width and height of resulting array
        # center_row, center_col: center position / optical axis
        # maxele:               maximum angle visible on resulting array

        # Azimuth is equal for all radial projections

        # create grid with only rows and cols
        rows = list(reversed(range(height)))
        cols = range(width)
        row, col = np.meshgrid(rows, cols, indexing='ij')
        #row, col = np.mgrid[:height,:width]
        # center rows and cols
        col = col - center_col
        row = row - center_row
        # calculate radius from center
        r = np.sqrt(col ** 2 + row ** 2)

        # calculate azimuth
        coords = coordinates.Coordinates3d(x=col, # x: column
                                         y=row, # y: reversed row (upwards)
                                         azimuth_offset = north_angle, # north angle
                                         azimuth_clockwise = clockwise # orientation
                                         )

        azimuth = coords.azimuth

        # mark outside regions when specified
        if not (maxele_row,maxele_col) == (None,None):
            # maximum radius from center / norming value
            norm = np.sqrt( (center_row - maxele_row) ** 2 + \
                            (center_col - maxele_col) ** 2 )

            azimuth = ma.masked_where(                                   \
                # condition: everything outside the maxele region
                r > norm,        \
                # array
                azimuth                                                  \
                )

        # return
        return azimuth


    # create a distortion map
    @staticmethod
    def distortionMap(in_coord, out_coord, method="nearest",
                      basedon="spherical"):
        """
        create a distortion map for fast distortion of images in 2d plane.
        This map can be used to distort efficiently with 
        scipy.ndimage.interpolation.map_coordinates(...)

        args:
            in_coord (Coordinates3d): input image coordinates
            out_coord (Coordinates3d): output image coordinates
            method (str): interpolation method, see scipy.interpolate.griddata
            basedon (str): base interpolation on spherical (azimuth, elevation)
                or carthesian (x,y) coordinates. Defaults to "spherical",
                because "carthesian" makes problems at too large numbers.

        returns:
            array of shape (shape(out_coord_2/out_coord_1),2) with interpolated 
            coordinates in input array. This array can directly be used as
            coordinate array for scipy.ndimage.interpolation.map_coordinates()
        """
        # input/output shape
        in_shape  = in_coord.shape
        out_shape = out_coord.shape

        # input image coordinates (row, col)
        in_row, in_col = np.mgrid[:in_shape[0],:in_shape[1]]
        in_row = in_row.flatten() # one dimension
        in_col = in_col.flatten() # one dimension
    
        if basedon == "carthesian":
            logger.debug("interpolate based on carthesian coordinates.")
            # input image coordinates (ele, azi)
            points = (in_coord.x.flatten(), 
                      in_coord.y.flatten())
            # output image coordinates (ele, azi)
            xi = (out_coord.x.flatten(),
                  out_coord.y.flatten())
        else:
            logger.debug("interpolate based on spherical coordinates.")
            # input image coordinates (ele, azi)
            points = (in_coord.azimuth.flatten(), 
                      in_coord.elevation.flatten())
            # output image coordinates (ele, azi)
            xi = (out_coord.azimuth.flatten(),
                  out_coord.elevation.flatten())
    
        logger.debug("interpolation started...")

        out_row_from_in = scipy.interpolate.griddata(
            points = points,
            values = in_row,
            xi     = xi,
            method = method
            )
        out_col_from_in = scipy.interpolate.griddata(
            points = points,
            values = in_col,
            xi     = xi,
            method = method
            )

        # stack row and col together
        distmap = np.dstack(
                # reshape to output shape
                # not exactly sure why this has to be transponated...
                (out_row_from_in.reshape(out_shape).T,
                 out_col_from_in.reshape(out_shape).T)
                 )

        logger.debug("interpolation ended!")

        # return the distortion map
        return DistortionMap(map=distmap, src_shape=in_shape) 




# class for distortionmaps
class DistortionMap(object):
    """
    class that holds a distortion map. Practically a subclass to ndarray.

    properties:
        map (array): the distortion map. If you set this, src_shape will be reset.
        src_shape (int tuple): shape of the input image used initially for the map
        out_shape (int tuple): shape of the output image when applying the map
    """
    def __init__(self, map, src_shape=None):
        """
        constructor

        args:
            map (array): distortion map, shape (shape(output),dim(input)) 
            src_shape (optional[int tuple]): shape of input image. If not
                specified, no tests can be performed if map is applied
        """
        self.map = map
        self.src_shape = src_shape

    ##################
    ### properties ###
    ##################
    # every attribute request (except _mapitself) goes directly to _map
    # this makes this class practically a subclass to ndarray
    def __getattr__(self, key):
        if key == '_map':
            raise AttributeError(" ".join([
                "Can't access _map attribute.",
                ]))
        return getattr(self._map, key)

    @property
    def map(self):
        return self._map

    @map.setter
    def map(self, newmap):
        self._map = newmap
        self.src_shape = None
        self.out_shape = np.shape(self.map)

    @property
    def out_shape(self):
        return self._out_shape

    @out_shape.setter
    def out_shape(self, newshape):
        self._out_shape = newshape

    @property
    def src_shape(self):
        return self._src_shape

    @src_shape.setter
    def src_shape(self, newshape):
        self._src_shape = newshape


