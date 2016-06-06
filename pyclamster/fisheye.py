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
            maxangle = np.pi / 2

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

    # create a distortion map
    def distortionMap(self, in_ele, in_azi, out_ele, out_azi, method="nearest"):
        """
        create a distortion map for fast distortion.
        This map can be used to distort efficiently with 
        scipy.ndimage.interpolation.map_coordinates(...)

        args:
            in_ele, in_azi (array_like): input image elevation and azimuth arrays
            out_ele, out_azi (array:like): output image elevation and azimuth arrays
            method (str): interpolation method, see scipy.interpolate.griddata

        returns:
            array of shape (shape(out_ele/out_azi),2) with interpolated 
            coordinates in input array. This array can directly be used as
            coordinate array for scipy.ndimage.interpolation.map_coordinates()
        """
        if not np.shape(in_ele) == np.shape(in_azi) or \
           not np.shape(out_ele) == np.shape(out_azi):
           raise ValueError("elevation/azimuth arrays have to have same shape!")

        # input/output shape
        in_shape = np.shape(in_ele)
        out_shape = np.shape(out_ele)

        # input image coordinates (row, col)
        in_row, in_col = np.mgrid[:in_shape[0],:in_shape[1]]
        in_row = in_row.reshape(np.prod(in_shape)) # one dimension
        in_col = in_col.reshape(np.prod(in_shape)) # one dimension
    
        # input image coordinates (ele, azi)
        points = (in_ele.reshape(np.prod(in_shape)), 
                  in_azi.reshape(np.prod(in_shape)))
        # output image coordinates (ele, azi)
        xi = (out_ele.reshape(np.prod(out_shape)),
              out_azi.reshape(np.prod(out_shape)))
    
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
                (out_col_from_in.reshape(out_shape),
                 out_row_from_in.reshape(out_shape))
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


###############
### Example ###
###############
if __name__ == '__main__':
    import numpy as np
    import scipy.interpolate
    import scipy.ndimage
    import image
    import os
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # read image
    img = image.Image(os.path.abspath(
        "../examples/images/stereo/Image_20160527_144000_UTCp1_4.jpg"
        ))
    # convert to grayscale
    #img.image = img.convert("L")
    # resize image
    img.image = img.resize((800,800))


    # image shape
    shape=np.shape(img.data)[:2]
    f=FisheyeProjection("equidistant")
    # create elevation and azimuth image coordinates
    ele=f.createFisheyeElevation(shape)
    azi=f.createAzimuth(shape,maxelepos=(int(shape[0]/2),0))
    # create distorted rect coordinates
    x=np.linspace(-20,20,num=300)
    y=np.linspace(-20,20,num=300)
    z=6 
    ele_rect=f.createRectElevation(x,y,z)
    azi_rect=f.createRectAzimuth(x,y)

    # create distortion map
    distmap = f.distortionMap(ele, azi, ele_rect, azi_rect, "nearest")

    # distort image
    distimage = img.applyDistortionMap(distmap)

    # plot results
    import matplotlib.pyplot as plt
    plt.subplot(321)
    plt.title("original image")
    plt.imshow(img, cmap="Greys_r", interpolation="nearest")
    plt.subplot(322)
    plt.title("rectified image")
    plt.imshow(distimage, cmap="Greys_r", interpolation="nearest")
    plt.subplot(323)
    plt.title("image elevation")
    plt.imshow(ele,interpolation="nearest")
    plt.colorbar()
    plt.subplot(325)
    plt.title("image azimuth")
    plt.imshow(azi,interpolation="nearest")
    plt.colorbar()
    plt.subplot(324)
    plt.title("rectified elevation")
    plt.imshow(ele_rect,interpolation="nearest")
    plt.colorbar()
    plt.subplot(326)
    plt.title("rectified azimuth")
    plt.imshow(azi_rect,interpolation="nearest")
    plt.colorbar()
    plt.show()
