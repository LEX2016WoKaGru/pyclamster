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
import copy

# External modules
import numpy as np
import numpy.ma as ma
import scipy.interpolate

# Internal modules


__version__ = "0.1"

# create logger
logger = logging.getLogger(__name__)


###############################
### classes for coordinates ###
###############################
class Coordinates3d(object):
    def __init__(self, dimnames, shape=None):
        # initialize base variables
        self._dim_names = dimnames
        # initialize shape
        self.shape = shape

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, newshape): # if the shape of the coordinates is set
        """
        set shape of coordinates
        args:
            newshape (tuple of int): new shape of coordinates. If newshape is
                None, all dimensions are set to None. If a reshape of the
                dimensions is possible, a reshape is performed on all
                dimensions. I a reshape is not possible, all dimensions
                are initialized with completely masked empty arrays of the
                new shape. If newshape is equal the old shape, do nothing.
        """
        self._shape = newshape # set new shape
        ### loop over all dimensions ###
        for dim in self._dim_names: # loop over all dimensions
            try:    shape = getattr(self, dim).shape # try to read shape
            except: shape = None # if not yet defined, use None

            if newshape is None: # newshape is None
                ### new shape is None --> set everything to None ###
                setattr(self, "_{}".format(dim), None)
                #logger.debug("newshape is None, setting {} to None".format(dim))
            else: # newshape is not None
                ### new shape is not None --> further investigation ###
                if np.prod(newshape) == np.prod(self.shape) and \
                    not shape is None and not newshape == self.shape:
                    ### reshape is possible --> reshape!  ###
                    # try to reshape current content
                    try:    
                        new = getattr(self, dim).reshape(newshape) # try
                        #logger.debug( " ".join([
                        #"reshaping {dim} from shape {shape}",
                        #"to shape {newshape}",
                        #]).format(dim=dim,newshape=newshape,shape=shape))
                    except: # dimension was not yet defined --> use empty
                        #logger.debug(" ".join([
                            #"can't reshape {dim} from shape {shape}",
                            #"to shape {newshape}.",
                            #"setting {dim} to empty array of shape {newshape}."
                            #]).format(dim=dim,newshape=newshape,shape=shape))
                        new = ma.masked_array(
                            data = np.empty(newshape),
                            mask = np.ones( newshape))
                        
                    # reshape variable
                    setattr(self, "_{}".format(dim), new)
                else: # reshape not possible
                    ### reshape NOT possible 
                    ### --> reinit with empty arrays if oldshape does not match
                    if shape != newshape: # only if new shape does not match
                        # set to an empty array
                        setattr(self, "_{}".format(dim), ma.masked_array(
                            data = np.empty(newshape),
                            mask = np.ones( newshape)))
                        #logger.debug( " ".join([
                        #"setting {dim} to completely masked array of shape",
                        #"{newshape} because shape {dimshape} didn't match",
                        #"newshape {newshape}."
                        #]).format(dim=dim,newshape=newshape,dimshape=shape))
        

    # set the coordinate to a new value
    def _set_coordinate(self, coord, value):
        """
        Set the coordinate 'coord' to value 'value'. The value is converted
        to an array or expanded to an array of appropriate shape if value
        only has length 1.
        If the other coordinates are undefined, set them to empty masked arrays
        of appropriate shape.

        args:
            coord (str): name of the coord attribute
            value (array_like or single numeric): new coordinate array. 
                Must be of shape self.shape. 
        """
        #logger.debug("attempt to set coordinate {} to {}.".format(coord,value))

        # find out names of remaining two dimensions
        i = self._dim_names.index(coord)
        otherdims = self._dim_names[:i] + self._dim_names[(i+1):]

        if not value is None: # if value is not None
            # make sure value is an array
            value = ma.asanyarray(value) # try to convert to array

            # check shape
            if not self.shape is None: # if shape is defined
                if np.prod(value.shape) == 1: # only one value was given
                    # filled constant array
                    value = np.full( self.shape, value, np.array(value).dtype)
                elif value.shape != self.shape: # value shape does not match
                    raise ValueError(
                    "invalid shape {} (not {}) of new coordinate {}".format(
                            value.shape, self.shape, coord))

            resval = value # this value

            # set other dims to completely masked array if necessary
            for dim in otherdims: # loop over all other dimensions
                try: dimval = getattr(self, dim) # try to read current dimval
                except: dimval = None # if not yet defined, use None
                if dimval is None: # if current dimval is not defined
                    setattr(self, "_{}".format(dim), ma.masked_array(
                        data = np.empty(self.shape),
                        mask = np.ones( self.shape)))

        else: # specified value is None
            if self.shape is None: # if no shape was defined yet
                resval = None # just set this dimension to None
            else: # shape is defined, set to empty array of appropriate shape
                resval =  ma.masked_array(
                    data = np.empty(self.shape),
                    mask = np.ones( self.shape))
                #logger.debug(
                  #"setting {} to completely masked arrays of shape {} ".format(
                        #",".join(self._dim_names),self.shape))

        # set resulting value
        #logger.debug("setting {} to {}".format(coord,resval))
        setattr(self, "_{}".format(coord), resval)

        try: # try this because resval can be None...
            if self.shape != resval.shape:
                #logger.debug("Adjusting shape from {} to {}".format(self.shape,
                    #resval.shape))
                self.shape = resval.shape
        except: pass

    # crop coordinates to a box
    def crop(self, box):
        """
        crop the coordinates in-place to a box
        args:
            box (4-tuple of int): (left, top, right, bottom)
        """

        for dim in self._dim_names: # loop over all dimensions
            new = getattr(self, dim)[box[1]:box[3], box[0]:box[2]]
            # set underlying coordinate directly
            setattr(self, "_{}".format(dim) , new)

    # cut out a box
    def cut(self, box):
        """
        cut the coordinates to a box and return it
        args:
            box (4-tuple of int): (left, top, right, bottom)
        return:
            coordinates = copied and copped instance
        """
        new = copy.deepcopy(self) # copy
        new.crop(box) # crop
        return new # return
        
            

# class for carthesian 3d coordinates
class CarthesianCoordinates3d(Coordinates3d):
    def __init__(self, 
                 shape=None,
                 x=None, y=None, z=None,
                 clockwise=False,
                 azimuth_offset=np.pi/2
                 ):
        """
        create a set of carthesian coordinates lying on a 2-dimensional grid.

        args:
            x,y,z (optional[array_like]): coordinates x, y and z
            clockwise(optional[boolean]): does the azimuth go clockwise? 
                Defaults to False (mathematical direction)
            azimuth_offset(optional[float]): azimuth angle offset (in radians). 
                The azimuth_offset is the angle between the positive x-axis
                and the 0-azimuth line depending on clockwise argument. 
                Defaults to 90°, pi/2, which is the top of the image at 
                counter-clockwise direction.

        returns:
            np.maskedarray of shape (width, height) with azimuth values
        """
        # parent constructor
        super().__init__(
            shape = shape,
            dimnames = ["x","y","z"]
            )

        # initially set underlying attributes to None
        self._x, self._y, self._z = (None, None, None)
        # copy over the arguments
        self.x = x
        self.y = y
        self.z = z
        self.clockwise = clockwise
        self.azimuth_offset = azimuth_offset

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

    # convert these carthesian coordinates to horizontal radius
    @property
    def radius_horz(self):
        """
        convert these carthesian coordinates to horizontal radius
        returns:
            an array with horizontal radius values
        """
        radius = np.sqrt( self.x ** 2 + self.y ** 2 )
        return radius

    # convert these carthesian coordinates to spherical elevation
    @property
    def elevation(self):
        """
        convert these carthesian coordinates to spherical elevation
        returns:
            an array with elevation values
        """
        return np.arctan( self.radius_horz / self.z )

    # convert these carthesian coordinates to spherical azimuth
    @property
    def azimuth(self):
        """
        convert these carthesian coordinates to spherical elevation
        returns:
            an array with azimuth values
        """
        north = self.azimuth_offset
        clockwise = self.clockwise

        north = - (north % (2*np.pi) )
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
        azimuth = np.arctan2(self.y, self.x) + 6 * np.pi + north

        # take azimuth modulo a full circle to have sensible values
        azimuth = azimuth % (2*np.pi)

        if clockwise: # turn around if clockwise
            azimuth = 2 * np.pi - azimuth

        return azimuth

    # convert these carthesian coordinates to spherical radius
    @property
    def radius(self):
        """
        convert these carthesian coordinates to spherical radius
        returns:
            an array with radius values
        """
        return np.sqrt( self.x ** 2 + self.y ** 2 + self.z ** 2 )

    # convert carthesian grid to polar grid
    def spherical(self, target=None, clockwise=None, azimuth_offset=np.pi/2):
        """
        convert these carthesian coordinates to spherical coordinates
        returns:
            an instance of class SphericalCoordinates3d
        """
        if not target is None:
            clockwise = target.clockwise
            azimuth_offset = target.azimuth_offset
        else:
            if clockwise is None:
                clockwise = self.clockwise
            if azimuth_offset is None:
                azimuth_offset = self.azimuth_offset

        # return coordinates
        return SphericalCoordinates3d(
            shape       = self.shape,
            azimuth     = self.azimuth, 
            elevation   = self.elevation,
            radius      = self.radius,
            clockwise   = clockwise,
            azimuth_offset = azimuth_offset
            )
                


# class for spherical 3d coordinates
class SphericalCoordinates3d(Coordinates3d):
    def __init__(self,
                 shape=None,
                 azimuth=None,
                 elevation=None,
                 radius=None,
                 clockwise=False,
                 azimuth_offset=np.pi/2
                 ):
        """
        create a set of spherical coordinates lying on a 2-dimensional grid.

        args:
            azimuth, elevation, radius (optional[array_like]): coordinates
            clockwise(optional[boolean]): does the azimuth go clockwise? 
                Defaults to False (mathematical direction)
            azimuth_offset(optional[float]): azimuth angle offset (in radians). 
                The azimuth_offset is the angle between the positive x-axis
                and the 0-azimuth line depending on clockwise argument. 
                Defaults to 90°, pi/2, which is the top of the image at 
                counter-clockwise direction.

        returns:
            np.maskedarray of shape (width, height) with azimuth values
        """
        # parent constructor
        super().__init__(
            shape=shape,
            dimnames = ["azimuth","elevation","radius"]
            )

        # copy over the arguments
        self.azimuth     = azimuth
        self.elevation   = elevation
        self.radius      = radius
        self.azimuth_offset = azimuth_offset
        self.clockwise = clockwise

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

    # convert spherical to carthesian x coordinate
    @property
    def x(self):
        """
        convert these spherical coordinates to carthesian x coordinate
        returns:
            an array of x values
        """
        return self.radius                          \
            * np.sin( self.elevation )              \
            * np.cos( self.azimuth + self.azimuth_offset )

    # convert spherical to carthesian y coordinate
    @property
    def y(self):
        """
        convert these spherical coordinates to carthesian y coordinate
        returns:
            an array of y values
        """
        return self.radius                              \
            * np.sin( self.elevation )                  \
            * np.sin( self.azimuth + self.azimuth_offset )

    # convert spherical to carthesian z coordinate
    @property
    def z(self):
        """
        convert these spherical coordinates to carthesian z coordinate
        returns:
            an array of z values
        """
        return self.radius * np.cos( self.elevation )

    # convert these spherical coordinates (elevation) to spherical radius
    # with given height z
    def radius_with_height(self, z):
        """
        convert these spherical coordinates (only elevation actually) 
        to the spherical radius given a height z
        returns:
            an array of radius values
        """
        return z / np.cos( self.elevation )

    # convert these spherical coordinates to carthesian coordinates
    def carthesian(self, target=None, clockwise=None, azimuth_offset=np.pi / 2):
        """
        convert these spherical coordinates to carthesian coordinates
        returns:
            an instance of class CarthesianCoordinates3d
        """
        if not target is None:
            clockwise = target.clockwise
            azimuth_offset = target.azimuth_offset
        else:
            if clockwise is None:
                clockwise = self.clockwise
            if azimuth_offset is None:
                azimuth_offset = self.azimuth_offset

        # return coordinates
        return CarthesianCoordinates3d(
            shape = self.shape,
            x     = self.x, 
            y     = self.y,
            z     = self.z,
            clockwise = clockwise,
            azimuth_offset = azimuth_offset
            )
                
