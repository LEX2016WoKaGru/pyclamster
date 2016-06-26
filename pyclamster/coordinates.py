# -*- coding: utf-8 -*-
"""
Created on 25.06.2016

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

# Internal modules
from . import utils


__version__ = "0.1"

# create logger
logger = logging.getLogger(__name__)


##########################
### Calculation Method ###
##########################
class CalculationMethod(object):
    def __init__(self,input,output,func):
        self.input  = set()
        self.output = set()
        if isinstance(input,str): self.input.add(input)
        else:                     self.input.update(input)
        if isinstance(output,str): self.output.add(output)
        else:                      self.output.update(output)

        self.func   = func

    # check if this calculation method can be applied on quantities
    def applicable(self,quantities):
        return all(x in quantities for x in self.input)

    # when this method is called
    def __call__(self):
        self.func() # call function

    # summary on stringification
    def __str__(self):
        lines = []
        lines.append("{func}".format(func=self.func))
        lines.append("{o} <- function of {i}".format(
            o=",".join(self.output),
            i=",".join(self.input)))
        return "\n".join(lines)


##############################
### Calculation Method set ###
##############################
class CalculationMethodSet(object):
    def __init__(self, *methods):
        self.methods = []
        for method in utils.flatten(methods):
            self.addmethod(method) 

    ################################################################
    ### make the method set behave correct in certain situations ###
    ################################################################
    # make it iterable
    def __iter__(self):
        try: del self.current # reset counter
        except: pass
        return self

    def __next__(self):
        try:    self.current += 1 # try to count up
        except: self.current  = 0 # if that didn't work, start with 0
        if self.current >= len(self.methods):
            del self.current # reset counter
            raise StopIteration # stop the iteration
        else:
            return self.methods[self.current]
        
    # make it indexable
    def __getitem__(self,key):
        return self.methods[key]

    # make it return something in boolean context
    def __bool__(self): # return value in boolean context
        return len(self.methods) > 0

    # make it callable
    def __call__(self): # when this set is called
        for method in self.methods: # loop over all methods
            method() # call the method


    # summary if converted to string
    def __str__(self):
        lines = []
        lines.extend(["==============================",
                      "| set of calculation methods |",
                      "=============================="])
        n = len(self.methods)
        if n > 0:
            lines.append("{n} calculation method{s}:".format(
                n=n,s='s' if n!=1 else ''))
            for i,method in enumerate(self.methods):
                lines.append("\nMethod Nr. {i}/{n}:\n{m}".format(i=i+1,
                    m=method,n=n))
        else:
            lines.append("no calculation methods\n")
        return "\n".join(lines)

        
    ###################################
    ### managing methods in the set ###
    ###################################
    def addmethod(self, method):
        self.methods.append(method)

    def add_new_method(self, input, output, func):
        # create new method
        method = CalculationMethod(input=input,output=output,func=func)
        self.addmethod(method) # add the method

    def removemethod(self, method):
        self.methods.remove(method)

    ############################################
    ### getting information from the methods ###
    ############################################
    # given a set of quantity names, determine which methods can be
    # applied DIRECTLY on them
    def applicable_methods(self, quantities):
        methods = CalculationMethodSet() # new empty set
        for method in self.methods: # loop over all methods
            # check if method is applicable and add it to list if yes
            if method.applicable(quantities):
                methods.addmethod(method)
        return methods

    # given a set of quantity names, determine which methods yield
    # any of these quantities DIRECTLY
    def methods_yield(self, quantities):
        methods = CalculationMethodSet() # new empty set
        for method in self.methods: # loop over all methods
            # check if method is applicable and add it to list if yes
            if any(q in method.output for q in quantities):
                methods.addmethod(method)
        return methods

    # return all calculatable quantities of this set
    @property
    def all_calculatable_quantities(self):
        q = set()
        for m in self: q.update(m.output)
        return q

    # return all needed quantities of this set
    @property
    def all_needed_quantities(self):
        q = set()
        for m in self: q.update(m.input)
        return q

    # given a set of quantity names, determine which other quantities can be
    # calculated based DIRECTLY on it
    def directly_calculatable_quantities(self, quantities):
        # get the applicable methods
        applicable = self.applicable_methods(quantities)
        # return all calculatable quantities
        return applicable.all_calculatable_quantities

    # given a set of quantity names, determine which other quantities can 
    # DIRECTLY calculate these quantities
    def quantities_can_calculate(self, quantities):
        # get all methods that yield any of the given quantities
        methods = self.methods_yield(quantities)
        # return all the needed quantities for this
        return methods.all_needed_quantities

    # given a set of quantity names, construct a calculation method set
    # with the correct order to calculate as much other quantities as possible
    def dependency_line(self, quantities):
        known_quantities = set(quantities) # copy of given quantities
        line = CalculationMethodSet() # new empty set
        while True:
            # get the applicable methods at this stage
            methods = self.applicable_methods(known_quantities)
            for method in list(methods): # loop over (a copy of) all methods
                if method.output.issubset(known_quantities): # if we know it
                    methods.removemethod(method) # don't consider it
            if methods: # if something can be calculated
                known_quantities.update(methods.all_calculatable_quantities)
                # extend the line with the methods
                for method in methods:
                    line.addmethod(method)
            else: break # nothing found, abort
        return line

            


###############################
### classes for coordinates ###
###############################
class BaseCoordinates3d(object):
    def __init__(self, dimnames, shape=None,clockwise=False,azimuth_offset=0):
        # initialize base variables
        self._dim_names = dimnames
        # initialize shape
        self.shape = shape
        self.clockwise = clockwise
        self.azimuth_offset = azimuth_offset

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
                elif np.prod(value.shape) == np.prod(self.shape):
                    # reshape
                    value = value.reshape(self.shape)
                elif value.shape != self.shape: # value shape does not match
                    raise ValueError(
                    "invalid shape {} (not {}) of new coordinate {}".format(
                            value.shape, self.shape, coord))
            else: # shape is not defined yet
                self.shape = value.shape # set it!

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
        
            


########################################
### convenient class for coordinates ###
########################################
class Coordinates3d(BaseCoordinates3d):
    def __init__(self, shape=None, azimuth_offset=0, clockwise=False,
                 center=None, **dimensions):

        # parent constructor
        super().__init__(
            shape=shape,
            dimnames = ['elevation','azimuth','radius','radiush','x','y','z'],
            azimuth_offset = azimuth_offset,
            clockwise = clockwise
            )

        self.center = center

        self.methods = CalculationMethodSet()
        # add methods
        self.methods.add_new_method(output='radius',input={'x','y','z'},
            func=self.radius_from_xyz)
        self.methods.add_new_method(output='radiush',input={'x','y'},
            func=self.radiush_from_xy)
        self.methods.add_new_method(output='radius',input={'radiush','z'},
            func=self.radius_from_radiush_z)
        self.methods.add_new_method(output='radius',input={'elevation',
            'radiush'}, func=self.radius_from_elevation_radiush)
        self.methods.add_new_method(output='azimuth',input={'x','y'},
            func=self.azimuth_from_xy)
        self.methods.add_new_method(output='elevation',input={'radiush','z'},
            func=self.elevation_from_radiush_z)
        self.methods.add_new_method(output='x',input={'azimuth','elevation',
            'radius'}, func=self.x_from_spherical)
        self.methods.add_new_method(output='x',input={'azimuth','radiush'},
            func=self.x_from_azimuth_radiush)
        self.methods.add_new_method(output='y',input={'azimuth','elevation',
            'radius'}, func=self.y_from_spherical)
        self.methods.add_new_method(output='y',input={'azimuth','radiush'},
            func=self.y_from_azimuth_radiush)
        self.methods.add_new_method(output='z',input={'azimuth','elevation',
            'radius'}, func=self.z_from_spherical)
        self.methods.add_new_method(output='x',input={'radiush','y'},
            func=self.x_from_radiush_y)
        self.methods.add_new_method(output='y',input={'radiush','x'},
            func=self.y_from_radiush_x)
        self.methods.add_new_method(output='z',input={'radius','radiush'},
            func=self.z_from_radiusses)
        self.methods.add_new_method(output='radiush',input={'elevation','z'},
            func=self.radiush_from_elevation_z)
        self.methods.add_new_method(output='radiush',input={'elevation',
            'radius'}, func=self.radiush_from_elevation_radius)
        self.methods.add_new_method(output='elevation',input={'radius',
            'radiush'}, func=self.elevation_from_radiusses)

        # fill with given dimensions
        if dimensions:
            self.fill(**dimensions)


    @property
    def x(self): 
        try: return self._x - self.center.x
        except:
            try: return self._x
            except: raise Exception("x is not specified yet")
    @property
    def y(self):
        try: return self._y - self.center.y
        except:
            try: return self._y
            except: raise Exception("y is not specified yet")
    @property
    def z(self): 
        try: return self._z - self.center.z
        except:
            try: return self._z
            except: raise Exception("z is not specified yet")
    @property
    def azimuth(self):   return self._azimuth
    @property
    def elevation(self): return self._elevation
    @property
    def radius(self):    return self._radius
    @property
    def radiush(self):    return self._radiush

    @x.setter
    def x(self, value):         self.fill(x=value)
    @y.setter
    def y(self, value):         self.fill(y=value)
    @z.setter
    def z(self, value):         self.fill(z=value)
    @azimuth.setter
    def azimuth(self, value):   self.fill(azimuth=value)
    @elevation.setter
    def elevation(self, value): self.fill(elevation=value)
    @radius.setter
    def radius(self, value):    self.fill(radius=value)
    @radiush.setter
    def radiush(self, value):   self.fill(radiush=value)

    @property
    def center(self): return self._center
    @center.setter
    def center(self, newcenter):
        if all(hasattr(newcenter,val) for val in ["x","y","z"]):
            self._center = newcenter

    # determine which dimensions are defined
    @property
    def defined_dimensions(self):
        defined = set()
        for dim in self._dim_names:
            isdefined = False
            try: value = getattr(self, dim)
            except:  pass
            if not value is None:
                try: isdefined = not value.mask.all()
                except AttributeError:
                    isdefined = True
            if isdefined:
                defined.add(dim)
        return(defined)
        

    # given specific values for some dimensions, calculate all others
    def fill_dependencies(self,dimensions):
        # get the depenency line
        dependency_line = self.methods.dependency_line(dimensions)

        # do everything in the dependency line
        dependency_line() # call 

    # set as much variables as you can based on given dimensions and already
    # defined dimensions
    def fill(self, **dimensions):
        #logger.debug("request to set {}".format(dimensions))
        # first, unset all dimensions that reverse-depend on the new dimensions
        for dim in dimensions.keys(): # loop over all new given dimensions
            # get all methods that yield this new dimension
            ms = set(); ms.add(dim)
            methods = self.methods.methods_yield(ms)
            #logger.debug("methods, that yield {}:".format(dim))
            for m in methods: # loop over all methods that yield this new dim
                # if for this method all information is already given
                if m.input.issubset(self.defined_dimensions):
                    # unset everything needed for this method
                    # because this is the reverse dependency of this new
                    # dimensions
                    for d in m.input:
                        #logger.debug(" ".join([
                        #"unsetting {d}, because {i} can calculate {dim}",
                        #"via {m}, and all of {i} are given."
                        #]).format(m=m,d=d,dim=dim,i=m.input))
                        self._set_coordinate(d, None)

                
        # initially set all given variables
        for dim, value in dimensions.items():
            #logger.debug("setting {} directly to value {}".format(dim,value))
            self._set_coordinate(dim, value)

        # create a set of defined dimensions, updated with new given dimensions
        merged = set(self.defined_dimensions)
        merged.update(dimensions.keys())

        # now fill the dependencies with all the information we have
        self.fill_dependencies(merged)
        
    ###########################
    ### calculation methods ###
    ###########################
    def radius_from_xyz(self):
        self._radius = np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def radiush_from_xy(self):
        self._radiush = np.sqrt(self.x**2 + self.y**2)

    def radius_from_radiush_z(self):
        self._radius = np.sqrt(self.radiush**2 + self.z**2)

    def radius_from_elevation_radiush(self):
        self._radius = self.radiush / np.sin( self.elevation )

    def azimuth_from_xy(self):
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

        self._azimuth = azimuth

    def elevation_from_radiush_z(self):
        self._elevation = np.tan(self.radiush / self.z)

    def x_from_spherical(self):
        self._x = self.radius                          \
            * np.sin( self.elevation )              \
            * np.cos( self.azimuth + self.azimuth_offset )

    def x_from_azimuth_radiush(self):
        self._x = self.radiush * np.cos( self.azimuth + self.azimuth_offset )

    def y_from_spherical(self):
        self._y = self.radius                              \
            * np.sin( self.elevation )                  \
            * np.sin( self.azimuth + self.azimuth_offset )

    def y_from_azimuth_radiush(self):
        self._y = self.radiush * np.sin( self.azimuth + self.azimuth_offset )

    def z_from_spherical(self):
        self._z = self.radius * np.cos( self.elevation )

    def x_from_radiush_y(self):
        self._x = np.sqrt(self.radiush**2 - self.y**2)

    def y_from_radiush_x(self):
        self._y = np.sqrt(self.radiush**2 - self.x**2)

    def z_from_radiusses(self):
        self._z = np.sqrt(self.radius**2 - self.radiush**2)

    def radiush_from_elevation_z(self):
        self._radiush = self.z * np.arctan( self.elevation )

    def radiush_from_elevation_radius(self):
        self._radiush = self.radius * np.sin( self.elevation )

    def elevation_from_radiusses(self):
        self._elevation = np.arcsin( self.radiush / self.radius )
    ###############################
    ### end calculation methods ###
    ###############################
    
    # summary when converted to string
    def __str__(self):
        formatstring = ["==================",
                        "| 3d coordinates |",
                        "=================="]
        formatstring.append("         shape: {}".format(self.shape))
        formatstring.append("     clockwise: {}".format(self.clockwise))
        formatstring.append("azimuth_offset: {}".format(self.azimuth_offset))
        formatstring.append("==================")
        for dim in self._dim_names:
            value = getattr(self, dim)
            isdefined = False
            if not value is None:
                try: isdefined = not value.mask.all()
                except AttributeError:
                    isdefined = True
            if isdefined: string = "defined"
            else:         string = "empty"
            formatstring.append("{:>11}: {}".format(dim,string))
        return("\n".join(formatstring))
