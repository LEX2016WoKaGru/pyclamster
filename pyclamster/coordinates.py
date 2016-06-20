# -*- coding: utf-8 -*-
"""
Created on 17.05.16

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
import numpy as np
import copy

# External modules

# Internal modules


__version__ = "0.1"

# create logger
logger = logging.getLogger(__name__)


def flatten(x):
    res = []                        # start with an empty list
    for v in x:                     # loop over all elements in given list
        if type(v) in (tuple,list): # if element is list-like
            res.extend(flatten(v))  # extend resulting list
        else:                       # if element is not list-like
            res.append(v)           # append the element
    return res # return resulting list
            

#################################################
### abstract classes for dependant quantities ###
#################################################
class DependantQuantity(object):
    def __init__(self, data=None, name=None):
        self.data = data
        self.name = name

    #############################################
    ### attributes/properties getters/setters ###
    #############################################
    # every attribute request (except _data itself) goes directly to _data
    # this makes this class practically a subclass to numpy.ndarray
    def __getattr__(self, key):
        if key == '_data':
            raise AttributeError(" ".join([
                "Can't access _data attribute.",
            ]))
        return getattr(self._data, key)

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self,newname):
        if not isinstance(newname,str):
            newname = self.__class__.__name__
        self._name = newname
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, newdata):
        self._data = np.ma.asanyarray(newdata)

    def __str__(self):
        return "{name}: {data}".format(name=self.name,data=self.data)

    # define own deepcopy mechanism
    def __deepcopy__(self, memo):
        new = self.__class__() # new empty instance
        new.__dict__ = copy.deepcopy(self.__dict__) # copy over dict
        return(new) # return


class DependantQuantitySet(object):
    def __init__(self, *quantities, shape=None):
        self.quantities = flatten(quantities)

    # make the quantities accessible as quasi-attributes
    def __getattr__(self, name):
        classes = [q.__class__.name for q in self._quantities]
        names   = [q.name           for q in self._quantities]
        if name in names: # check if name matches
            return self._quantities[names.index(name)]
        elif name in classes: # check if it looks like classname
            return self._quantities[classes.index(name)]
        else: # default behavious
            raise AttributeError
            
    
    # make the quantity set iterable
    def __iter__(self):
        return self

    def __next__(self):
        try:    self.current += 1 # try to count up
        except: self.current  = 0 # if that didn't work, start with 0
        if self.current >= len(self.quantities):
            del self.current # reset counter
            raise StopIteration # stop the iteration
        else:
            return self.quantities[self.current]

    def _check_shape(self,newshape):
        # check if newshape is compatible with current shape
        for quantity in self.quantities:
            # if shape does not match
            if np.prod(newshape) != np.prod(quantity.shape):
                raise Exception(" ".join([
                    'New shape {newshape} is incompatible with shape',
                    '{othershape} of quantity {quantity}.',
                    'You may set the set shape to None to empty all',
                    'dependant quantities in the set and then set the',
                    'shape to your desired value.'
                    ]).format(quantity=quantity.name, 
                        othershape=quantity.shape, newshape=newshape))
        return True


    @property
    def shape(self):
        try:    return self._shape
        except: return None

    @shape.setter
    def shape(self, newshape):
        if self.shape is None: # if no shape is set
            for quantity in self.quantities:
                quantity.data = np.ma.masked_array(np.empty(newshape),
                    np.ones(newshape))
        elif newshape is not None:
            self._check_shape(newshape) # check if shape matches
    
            # set new shape to all quantities
            for quantity in self.quantities:
                if np.prod(quantity.shape) == 1: # only one value
                    quantity.data = np.full(shape,quantity.data)
                else: # multiple values
                    quantity.data = quantity.reshape(newshape)
        else: # new shape is None
            # set everything to empty arrays
            for quantity in self.quantities:
                quantity.data = np.array(())

        # set internal attribute
        self._shape = newshape
            
                

    @property
    def quantities(self):
        try:    return self._quantities
        except: return []

    @quantities.setter
    def quantities(self, newquantities):
        domshape = ()
        # check if shapes are compatible
        for quantity in newquantities:
            # higher dimensions dominate
            if len(quantity.shape) > len(domshape):
                domshape = quantity.shape
                
            if np.prod(quantity) == 1: continue # never mind if only 1 element
            # if shape does not match
            if np.prod(domshape) != np.prod(quantity.shape):
                raise ValueError(" ".join([
                    'shape {newshape} of new quantity {quantity} is',
                    'incompatible with set shape {domshape}',
                    ]).format(quantity=quantity.name, domshape=domshape,
                    newshape=quantity.shape))

            self.shape = domshape

        # if you reach this point, the new quantities are okay
        self._quantities = newquantities
        self.shape = domshape
                
        


    def addquantity(self, quantity):
        newquantities = copy.deepcopy(self.quantities) # copy old quantities
        for q in self.quantities:
            if quantity == q: # quantity already registered
                raise ValueError(" ".join(["Quantity {q} is already",
                "present in the set."]).format(q=quantity.name))
        newquantities.append(quantity)  # append new quantity
        self.quantities = newquantities # set new quantities

    def removequantity(self, quantity):
        try: # try to find index
            indices = [i for i,q in enumerate(self.quantities) \
                if q == quantity]
        except ValueError: # didn't work, try to remove based on class
            quantityclasses = [q.__class__ for q in self.quantities]
            try:    
                indices = [i for i,q in enumerate(quantityclasses) \
                    if q == quantity]
            except: 
                raise ValueError("{} is not in the set.".format(quantity))
        newquantities = copy.deepcopy(self.quantities) # copy old quantities
        for index in sorted(indices,reverse=True): # loop in REVERSED order
            del newquantities[index] # remove found elements
        self.quantities = newquantities # set new quantities

    # check if this set includes a given quantity
    def hasquantity(self, quantity):
        return quantity in list(self.quantities)

    # summary if converted to string
    def __str__(self):
        lines = []
        lines.extend(["===============================",
                      "| set of dependant quantities |",
                      "==============================="])
        if len(self.quantities) == 0: # no quantities
            lines.append("no quantities registered.")
        else: # quantities specified
            lines.append("{} quantities:".format(len(self.quantities)))
            for quantity in self.quantities:
                lines.append("---------- {} ----------".format(quantity.name))
                if max(quantity.shape) >= 6:
                    lines.append("shape {}".format(quantity.shape))
                else:
                    lines.append("{}".format(quantity.data))
    
        return "\n".join(lines)


class CalculationMethod(object):
    def __init__(self,input,output):
        self.input = input
        self.output = output

    # chack if this calculation method can be applied on a given quantity set
    def applicable(self,quantities):
        return all(quantities.hasquantity(i) for i in self.input)
                
            
        

class CalculationMethodSet(object):
    def __init__(self, *methods):
        self.methods = []
        for method in flatten(methods):
            self.addmethod(method) 

    def __iter__(self):
        return self

    def __next__(self):
        try:    self.current += 1 # try to count up
        except: self.current  = 0 # if that didn't work, start with 0
        if self.current >= len(self.methods):
            del self.current # reset counter
            raise StopIteration # stop the iteration
        else:
            return self.methods[self.current]

    def addmethod(self, method):
        self.methods.append(method)

    def removemethod(self, method):
        self.methods.remove(method)

    def yields_something_given(self, quantities):
        pass

    def applicable_methods(self, quantities):
        methods = CalculationMethodSet()
        for method in self.methods: # loop over all methods
            # check if method is applicable and add it to list if yes
            if method.applicable(quantities):
                methods.addmethod(quantities)

    # summary if converted to string
    def __str__(self):
        lines = []
        lines.extend(["==============================",
                      "| cet of calculation methods |",
                      "=============================="])
        lines.append("calculation methods:")
        for method in self.methods:
            lines.append("{}".format(method))

        return "\n".join(lines)


###########################
### calculation methods ###
###########################
#class radius_from_xyz(CalculationMethod):
    #pass
    #_radius = np.sqrt(x**2 + y**2 + z**2)

class x_from_spherical(CalculationMethod):
    pass
#    _x = radius                          \
#        * np.sin( elevation )              \
#        * np.cos( azimuth + azimuth_offset )
#
class x_from_azimuth_radiush(CalculationMethod):
    pass
#    _x = radiush * np.cos( azimuth + azimuth_offset )
#
class x_from_radiush_y(CalculationMethod):
    pass
#    _x = np.sqrt(radiush**2 - y**2)
#
#class radiush_from_xy(CalculationMethod):
#    _radiush = np.sqrt(x**2 + y**2)
#
#class radius_from_radiush_z(CalculationMethod):
#    _radius = np.sqrt(radiush**2 + z**2)
#
#class radius_from_elevation_radiush(CalculationMethod):
#    _radius = radiush / np.sin( elevation )
#
#class azimuth_from_xy(CalculationMethod):
#    north = azimuth_offset
#    clockwise = clockwise
#
#    north = - (north % (2*np.pi) )
#    if clockwise:
#        north = - north
#
#    # note np.arctan2's way of handling x and y arguments:
#    # np.arctan2( y, x ), NOT np.arctan( x, y ) !
#    #
#    # np.arctan2( y, x ) returns the SIGNED (!)
#    # angle between positive x-axis and the vector (x,y)
#    # in radians
#
#    # the azimuth angle is...
#    # ...the SIGNED angle between positive x-axis and the vector...
#    # ...plus some full circle to only have positive values...
#    # ...minux angle class as "CalculationMethod" (modulo 2*pi to be precise)
#    # -->  azi is not angle to x-axis but to NORTH
#    azimuth = np.arctan2(y, x) + 6 * np.pi + north
#
#    # take azimuth modulo a full circle to have sensible values
#    azimuth = azimuth % (2*np.pi)
#
#    if clockwise: # turn around if clockwise
#        azimuth = 2 * np.pi - azimuth
#
#    _azimuth = azimuth
#
#class elevation_from_radiush_z(CalculationMethod):
#    _elevation = np.tan(radiush / z)
#
#class y_from_spherical(CalculationMethod):
#    _y = radius                              \
#        * np.sin( elevation )                  \
#        * np.sin( azimuth + azimuth_offset )
#
#class y_from_azimuth_radiush(CalculationMethod):
#    _y = radiush * np.sin( azimuth + azimuth_offset )
#
#class z_from_spherical(CalculationMethod):
#    _z = radius * np.cos( elevation )
#
#class y_from_radiush_x(CalculationMethod):
#    _y = np.sqrt(radiush**2 - x**2)
#
#class z_from_radiusses(CalculationMethod):
#    _z = np.sqrt(radius**2 - radiush**2)
#
#class radiush_from_elevation_z(CalculationMethod):
#    _radiush = z * np.arctan( elevation )
#
#class radiush_from_elevation_radius(CalculationMethod):
#    _radiush = radius * np.sin( elevation )
#
#class elevation_from_radiusses(CalculationMethod):
#    _elevation = np.arcsin( radiush / radius )

###################################################
### classes for spherical/carthesian quantities ###
###################################################
class x(DependantQuantity):
    pass

class y(DependantQuantity):
    pass

class z(DependantQuantity):
    pass

class elevation(DependantQuantity):
    pass

class azimuth(DependantQuantity):
    pass

class radius(DependantQuantity):
    pass

class radiush(DependantQuantity):
    pass

class CarthesianSphericalVariables(DependantQuantitySet):
    pass

class CarthesianSphericalSystem(CarthesianSphericalVariables):
    pass
