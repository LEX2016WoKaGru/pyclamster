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

# Internal modules


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
                               maxangle=None,
                               maxanglepos=None,
                               projection=None
                               ):
        """
        create a np.ndarray that holds an elevation for each pixel of an image
        according to fisheye projection type

        args:
            shape(2-tuple of int): shape of image (width, height)
            center(optional[2-tuple of int]): center/optical axis position on image (row,col).
                if not specified, the center of the image is assumed.
            maxanglepos(optional[2-tuple of int]): position of pixel with maxangle visible (row,col).
                if not specified, the left center of the image is assumed.
            maxangle(optional[float]): the maximum angle (rad) from optical axis/center visible on the image

        returns:
            np.maskedarray of shape (width, height) with elevation values
        """
        # shape
        try:
            width, height = shape
        except:
            raise ValueError("shape argument not defined as 2-tuple.")

        # center
        try:
            center_row, center_col = center
        except:
            logger.debug("center not specified as 2-tuple, assuming image center")
            center_row, center_col = int(height/2),int(width/2) # image center

        # maxangle
        if maxangle is None:
            logger.debug("maxangle not defined, assuming 90 degrees.")
            maxangle = np.pi / 4

        # maxangle pixel position
        try:
            maxangle_row, maxangle_col = maxanglepos
        except:
            logger.debug("maxangle position not specified as 2-tuple, assuming left image center")
            maxangle_row, maxangle_col = int(height/2),int(0) # left center


        # projection
        if projection is None: # projection defined?
            projection = self.projection # default projection from object
        if projection is None: # if projection is still not defined
            raise ValueError("no fisheye projection specified.")

            
        # width, height:          width and height of resulting array
        # center_row, center_col: center position / optical axis
        # maxangle:               maximum angle visible on resulting array
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
            norm = np.sqrt( (center_row - maxangle_row) ** 2 + \
                            (center_col - maxangle_col) ** 2 )

            # norm the radius to the maximum angle
            elevation = r / norm * maxangle 

            
        else:
            raise ValueError(" ".join((
                "unimplemented fisheye projection '{}'",
                "You should never see this...")
                ).format(projection))

        # mask everything outside the sensible region
        elevation = ma.masked_where(                                   \
            # condition: everything outside the maxangle region
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
                      clockwise=True,
                      north_angle=0,
                      ):
        """
        create a np.ndarray that holds an azimuth for each pixel of an image

        args:
            shape(2-tuple of int): shape of image (width, height)
            center(optional[2-tuple of int]): center/optical axis position on image (row,col).
                if not specified, the center of the image is assumed.
            maxelepos(optional[2-tuple of int]): position of pixel with max elevation angle visible (row,col).
                if not specified, the left center of the image is assumed.
            clockwise(optional[boolean]): does the azimuth go clockwise? Defaults to True.
            north_angle(optional[float]): north angle on the image (rad). Defaults to 0.

        returns:
            np.maskedarray of shape (width, height) with azimuth values
        """
        # shape
        try:
            width, height = shape
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

        # Azimuth equal for all radial projections

        # create grid with only rows and cols
        row, col = np.mgrid[:height, :width]
        # center rows and cols
        col = col - center_col
        row = row - center_row
        # if clockwise azimuth, invert x
        if clockwise: col = -col
        # calculate radius from center
        r = np.sqrt(col ** 2 + row ** 2)

        # calculate azimuth
        azimuth = np.arctan2(col, -row) + np.pi + north_angle
        azimuth[azimuth > 2 * np.pi] = azimuth[azimuth > 2 * np.pi] - 2 * np.pi

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

    # create an elevation matrix on a rectangular grid of x,y,z coordinates
    def createRectElevation(self,x,y,z):
        """
        create an elevation matrix on a rectangular grid of x,y,z coordinates

        args:
            x,y (array_like): coordinate sequences of rectangular x,y grid
            z (float): height

        returns:
            an array of shape(len(y),len(x)) with elevation values    
        """
        xi, yi = np.meshgrid(x,y)
        r = np.sqrt( xi ** 2 + yi ** 2 )
        ele = np.arctan( r / z )
        return ele

    # create an azimuth matrix on a rectangular grid of x,y coordinates
    def createRectAzimuth(self,x,y,north_angle=0):
        """
        create an azimuth matrix on a rectangular grid of x,y coordinates

        args:
            x,y (array_like): coordinate sequences of rectangular x,y grid
            north_angle (float): north angle 

        returns:
            an array of shape(len(y),len(x)) with azimuth values    
        """
        xi, yi = np.meshgrid(x,y)
        azi= np.arctan2(-xi, -yi) + np.pi + north_angle
        azi[azi > 2 * np.pi] = azi[azi > 2 * np.pi] - 2 * np.pi
        return azi



###############
### Example ###
###############
if __name__ == '__main__':
    import numpy as np
    import scipy.interpolate
    import image
    import os
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # read image
    img = image.Image(os.path.abspath("../examples/images/stereo/Image_20160527_144000_UTCp1_3.jpg"))

    # image shape
    shape=np.shape(img.data)[:2];
    f=FisheyeProjection("equidistant");
    # create elevation and azimuth image coordinates
    ele=f.createFisheyeElevation(shape);
    azi=f.createAzimuth(shape,maxelepos=(int(shape[0]/2),0));
    # create distorted rect coordinates
    x=np.arange(-3,4);
    y=np.arange(-2,3);z=10;
    ele_rect=f.createRectElevation(x,y,z);
    azi_rect=f.createRectAzimuth(x,y)

    # plot results
    import matplotlib.pyplot as plt
    plt.subplot(221)
    plt.title("image elevation")
    plt.imshow(ele,interpolation="nearest")
    plt.colorbar()
    plt.subplot(222)
    plt.title("image azimuth")
    plt.imshow(azi,interpolation="nearest")
    plt.colorbar()
    plt.subplot(223)
    plt.title("distorted elevation")
    plt.imshow(ele_rect,interpolation="nearest")
    plt.colorbar()
    plt.subplot(224)
    plt.title("distorted azimuth")
    plt.imshow(azi_rect,interpolation="nearest")
    plt.colorbar()
    plt.show()
