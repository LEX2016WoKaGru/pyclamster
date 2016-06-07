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
                      north_angle=0,
                      ):
        """
        create a np.ndarray that holds an azimuth for each pixel of an image

        args:
            shape(2-tuple of int): shape of image (height, width)
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
        azimuth = self.carthesian2azimuth( 
                        col,                 # x: column
                        row,                 # y: reversed row (upwards)
                        north_angle,         # north angle
                        clockwise            # clockwise or anticlockwise
                        )

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

    ##################################
    ### Coordinate transformations ###
    ##################################
    # create an elevation matrix on a rectangular grid of x,y,z coordinates
    def carthesian2elevation(self,x,y,z):
        """
        create an elevation matrix on a rectangular grid of x,y,z coordinates
        args:
            x,y (array_like): x and y coordinates of rectangular grid (use e.g. np.meshgrid(xseq,yseq))
            z (float): height
        returns:
            an array of shape(len(y),len(x)) with elevation values    
        """
        r = np.sqrt( x ** 2 + y ** 2 )
        ele = np.arctan( r / z )
        return ele

    # create an azimuth matrix on a rectangular grid of x,y coordinates
    def carthesian2azimuth(self,x,y,north_angle=0,clockwise=False):
        """
        create an azimuth matrix on a rectangular grid of x,y coordinates
        args:
            x,y (array_like): x and y coordinates of rectangular grid 
                (use e.g. np.meshgrid(xseq,yseq))
            north_angle (optional[float]): north angle - 0 azimuth angle offset from 
                positive x-axis. If clockwise is True, the angle is counted
                negative. Defaults to 0 (right side of image)
            clockwise (optional[bool]): clockwise angle rotation? Defaults to
                False, mathematical definition of angle.
        returns:
            an array of shape(len(y),len(x)) with azimuth values    
        """
        north = - (north_angle % (2*np.pi) )
        if clockwise:
            north = - north

        # note np.arctan2's way of handling x and y arguments:
        # np.arctan2( y, x ), NOT np.arctan( x, y ) !
        #
        # np.arctan2( y, x ) returns the SIGNED (!)
        # angle between positive x-axis and the vector (x,y)
        # in radians

        # the azimuth angle is...
        # ...the SIGNED angle between positive x-axis and the vector...
        # ...plus some full circle to only have positive values...
        # ...minux angle defined as "NORTH" (modulo 2*pi to be precise)
        # -->  azi is not angle to x-axis but to NORTH
        azi = np.arctan2(y, x) + 6 * np.pi + north

        # take azimuth modulo a full circle to have sensible values
        azi = azi % (2*np.pi)

        if clockwise: # turn around if clockwise
            azi = 2 * np.pi - azi

        return azi

    # convert carthesian grid to polar r
    def carthesian2r(self, x, y, z):
        return np.sqrt( x ** 2 + y ** 2 + z ** 2 )

    # convert polar grid to carthesian x coordinate
    def polar2x(self, azi, ele, r, north_angle):
        """
        convert polar grid to carthesian x coordinate
        args:
            azi,ele,r (array_like): azimuth, elevation and radius
            north_angle(optional[float]): north angle on the image (rad). Defaults to 0.
        returns:
            an array with shape of azi, ele and r with x values    
        """
        return r * np.sin( ele ) * np.cos( azi + north_angle )

    # convert polar grid to carthesian y coordinate
    def polar2y(self, azi, ele, r, north_angle):
        """
        convert polar grid to carthesian y coordinate
        args:
            azi,ele,r (array_like): azimuth, elevation and radius
            north_angle(optional[float]): north angle on the image (rad). Defaults to 0.
        returns:
            an array with shape of azi, ele and r with z values    
        """
        return r * np.sin( ele ) * np.sin( azi + north_angle )

    # convert polar grid to carthesian y coordinate
    def polar2z(self, ele, r):
        """
        convert polar grid to carthesian z coordinate
        args:
            azi,ele,r (array_like): azimuth, elevation and radius
        returns:
            an array with shape of azi, ele and r with z values    
        """
        return r * np.cos( ele )

    def polarwithheight2r(self, ele, z):
        return z / np.cos( ele )

    # convert polar grid to carthesian grid
    def polar2carthesian(self, azi, ele, r, north_angle):
        return (
                self.polar2x(azi, ele, r, north_angle), 
                self.polar2y(azi, ele, r, north_angle),
                self.polar2z(ele, r)
                )

    # convert carthesian grid to polar grid
    def carthesian2polar(self, x, y, z, north_angle=0, clockwise=False):
        return (
                self.carthesian2azimuth(x,y,north_angle,clockwise), 
                self.carthesian2elevation(x,y,z),
                self.carthesian2r(x,y,z)
                )

    # create a distortion map
    def distortionMap(self, in_coord_1, in_coord_2, out_coord_1, out_coord_2, method="nearest"):
        """
        create a distortion map for fast distortion.
        This map can be used to distort efficiently with 
        scipy.ndimage.interpolation.map_coordinates(...)

        args:
            in_coord_2, in_coord_1 (array_like): input image coordinate arrays (e.g. azimuth/elevation or x/y)
            out_coord_2, out_coord_1 (array:like): output image coordinate arrays (e.g. azimuth/elevation or x/y)
            method (str): interpolation method, see scipy.interpolate.griddata

        returns:
            array of shape (shape(out_coord_2/out_coord_1),2) with interpolated 
            coordinates in input array. This array can directly be used as
            coordinate array for scipy.ndimage.interpolation.map_coordinates()
        """
        if not np.shape(in_coord_2) == np.shape(in_coord_1) or \
           not np.shape(out_coord_2) == np.shape(out_coord_1):
           raise ValueError("elevation/azimuth arrays have to have same shape!")

        # input/output shape
        in_shape = np.shape(in_coord_1)
        out_shape = np.shape(out_coord_1)

        # input image coordinates (row, col)
        in_row, in_col = np.mgrid[:in_shape[0],:in_shape[1]]
        in_row = in_row.reshape(np.prod(in_shape)) # one dimension
        in_col = in_col.reshape(np.prod(in_shape)) # one dimension
    
        # input image coordinates (ele, azi)
        points = (in_coord_1.reshape(np.prod(in_shape)), 
                  in_coord_2.reshape(np.prod(in_shape)))
        # output image coordinates (ele, azi)
        xi = (out_coord_1.reshape(np.prod(out_shape)),
              out_coord_2.reshape(np.prod(out_shape)))
    
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


###############################
### classes for coordinates ###
###############################
class Coordinates3d(object):
    def __init__(self, dimnames):
        # initialize base variables
        self._dim_names = dimnames

    # set the coordinate to a new value
    def _set_coordinate(self, coord, value):
        """
        Set the coordinate 'coord' to value 'value'.
        If the other coordinates are undefined, set them to empty masked arrays
        of appropriate shape.

        args:
            value (array_like): new coordinate array. Must have 
                the same shape as other two coordinate dimensions (if defined).
        """
        # find out names of remaining two dimensions
        i = self._dim_names.index(coord)
        otherdims = self._dim_names[:i] + self._dim_names[(i+1):]

        try: # test if value is some kind of array
            value.shape
        except: # if not...
            if not value is None: # if value is not None
                try:    value = np.asarray(value) # try to convert to array
                except: pass

        # check if shape matches
        if not value is None: # only if something was specified
            try:
                for dim in otherdims:
                    if not getattr(self,dim) is None:
                        if not value.shape == getattr(self,dim).shape:
                            raise ValueError(
                              "shape of new {} does not match {} shape".format(
                                  coord,dim))
            except:
                raise ValueError("new {} coordinate is not array-like!".format(
                    coord))

            # set the underlying attribute
            setattr(self,"_{}".format(coord), value)

            # set other dims to completely masked array if necessary
            for dim in otherdims:
                if getattr(self, dim) is None:
                    setattr(self, dim, ma.masked_array(
                        data = np.empty(value.shape),
                        mask = np.ones( value.shape)))


# class for carthesian 3d coordinates
class CarthesianCoordinates3d(Coordinates3d):
    def __init__(self, x=None, y=None, z=None):
        # parent constructor
        super().__init__(dimnames = ["x","y","z"])

        # initially set underlying attributes to None
        self._x, self._y, self._z = (None, None, None)
        # copy over the arguments
        self.x = x
        self.y = y
        self.z = z

    @property
    def x(self): return self._x
    @property
    def y(self): return self._y
    @property
    def z(self): return self._z

    @x.setter
    def x(self, value): self._set_coordinate("x", value)
    @y.setter
    def y(self, value): self._set_coordinate("y", value)
    @z.setter
    def z(self, value): self._set_coordinate("z", value)


# class for spherical 3d coordinates
class SphericalCoordinates3d(Coordinates3d):
    def __init__(self, azimuth=None, elevation=None, radius=None):
        # parent constructor
        super().__init__(dimnames = ["azimuth","elevation","radius"])

        # initially set underlying attributes to None
        self._azimuth, self._elevation, self._radius = (None, None, None)
        # copy over the arguments
        self.azimuth   = azimuth
        self.elevation = elevation
        self.radius    = radius

    @property
    def azimuth(self):   return self._azimuth
    @property
    def elevation(self): return self._elevation
    @property
    def radius(self):    return self._radius

    @azimuth.setter
    def azimuth(self, value):   self._set_coordinate("azimuth", value)
    @elevation.setter
    def elevation(self, value): self._set_coordinate("elevation", value)
    @radius.setter
    def radius(self, value):    self._set_coordinate("radius", value)


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

    # read an image
    img = image.Image(os.path.abspath(
        "../examples/images/stereo/Image_20160527_144000_UTCp1_4.jpg"
        ))
    # convert to grayscale
    #img.image = img.convert("L")
    # resize image
    #img.image = img.resize((800,800))

    ### create a fisheye projection object ###
    f=FisheyeProjection("equidistant")

    ### create rectified coordinates ###
    outshape=(500,500) # size of output image
    rect_north_angle = np.pi / 2 # north angle of rectified image
    rect_x,rect_y=np.meshgrid(
        np.linspace(-20,20,num=outshape[1]),# image x coordinate goes right
        np.linspace(20,-20,num=outshape[0]) # image y coordinate goes up
        )
    rect_z = 5 # rectify for height rect_z
    rect_azi, rect_ele, rect_r = f.carthesian2polar(
        rect_x, rect_y, rect_z,
        north_angle = rect_north_angle,
        clockwise=False
        )

    ### create spherical coordinates of original image ###
    shape=np.shape(img.data)[:2] # shape of image
    orig_north_angle = 7 * np.pi / 4 # north angle of original image
    center = None # center of elevation/azimuth in the image
    maxelepos = (0,int(shape[1]/2)) # (one) position of maxium elevation
    maxele = np.pi / 2.3 # maximum elevation on the image border, < 90° here

    orig_ele=f.createFisheyeElevation(
        shape,
        maxelepos=maxelepos,
        maxele=maxele,
        center=center
        )
    orig_azi=f.createAzimuth(
        shape,
        maxelepos=maxelepos,
        center=center,
        north_angle = orig_north_angle,
        clockwise=False
        )
    orig_r = f.polarwithheight2r(ele=orig_ele, z=rect_z)

    ### create rectified coordinates of original image azimuth/elevation ###
    orig_x, orig_y, orig_z  = f.polar2carthesian(orig_azi,orig_ele,orig_r,rect_north_angle)


    ### create rectification map ###
    # based on regular grid
    logger.debug("calculating rectification map")
    distmap = f.distortionMap(orig_x, orig_y, rect_x, rect_y, "nearest")
    # based directly on azimuth and elevation
    #distmap = f.distortionMap(orig_azi, orig_ele, rect_azi, rect_ele, "nearest")

    ### rectify image ##
    rectimage = img.applyDistortionMap(distmap)

    ### plot results ###
    import matplotlib.pyplot as plt
    plt.subplot(3,4,1)
    plt.title("original image (fix)")
    plt.imshow(img.data, interpolation="nearest")
    plt.subplot(3,4,2)
    plt.title("image radius (calculated)")
    plt.imshow(orig_r, interpolation="nearest")
    plt.colorbar()
    plt.subplot(3,4,3)
    plt.title("rectified r (calculated)")
    plt.imshow(rect_r,interpolation="nearest")
    plt.colorbar()
    plt.subplot(3,4,4)
    plt.title("rectified image (calculated)")
    plt.imshow(rectimage.data, interpolation="nearest")
    plt.subplot(3,4,5)
    plt.title("image elevation (fix)")
    plt.imshow(orig_ele,interpolation="nearest")
    plt.colorbar()
    plt.subplot(3,4,9)
    plt.title("image azimuth (fix)")
    plt.imshow(orig_azi,interpolation="nearest")
    plt.colorbar()
    plt.subplot(3,4,6)
    plt.title("image x (calculated)")
    plt.imshow(orig_x,interpolation="nearest")
    plt.colorbar()
    plt.subplot(3,4,10)
    plt.title("image y (calculated)")
    plt.imshow(orig_y,interpolation="nearest")
    plt.colorbar()
    plt.subplot(3,4,7)
    plt.title("rectified x (fix)")
    plt.imshow(rect_x,interpolation="nearest")
    plt.colorbar()
    plt.subplot(3,4,11)
    plt.title("rectified y (fix)")
    plt.imshow(rect_y,interpolation="nearest")
    plt.colorbar()
    plt.subplot(3,4,8)
    plt.title("rectified elevation (calculated)")
    plt.imshow(rect_ele,interpolation="nearest")
    plt.colorbar()
    plt.subplot(3,4,12)
    plt.title("rectified azimuth (calculated)")
    plt.imshow(rect_azi,interpolation="nearest")
    plt.colorbar()
    plt.show()
