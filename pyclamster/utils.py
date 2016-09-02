# -*- coding: utf-8 -*-
"""
Created on 25.06.16

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

# Internal modules


__version__ = "0.1"

# create logger
logger = logging.getLogger(__name__)

# recursive list flattening
def flatten(x):
    res = []                        # start with an empty list
    for v in x:                     # loop over all given list elements
        if type(v) in (tuple,list): # if element is list-like
            res.extend(flatten(v))  # extend resulting list
        else:                       # if element is not list-like
            res.append(v)           # append the element
    return res # return resulting list
            

##################
### conversion ###
##################
def deg2rad(x):
    """
    convert degrees to radians
    args:
        x(numeric): angle in radians
    returns:
        radianangle = numeric
    """
    return x / 360 * 2 * np.pi

def rad2deg(x):
    """
    convert radians to degrees
    args:
        x(numeric): angle in degrees
    returns:
        degreeangle = numeric
    """
    return x / ( 2 * np.pi ) * 360

def pos_rad(x):
    """
    convert the given radian angle to a positive value
    args:
        x(numeric): angle in radians
    return:
        radianangle = numeric
    """
    return (x%(2*np.pi)+2*np.pi)%(2*np.pi)

def shift_matrix(dimbase, dimshift, rowshift, colshift):
    NROWSHIFT = dimshift[0]
    NROWBASE = dimbase[0]
    NCOLSHIFT = dimshift[1]
    NCOLBASE = dimbase[1]
    ROWDIFF = (NROWBASE - NROWSHIFT) / 2
    COLDIFF = (NCOLBASE - NCOLSHIFT) / 2
    MAXROW = (NROWBASE + NROWSHIFT) / 2
    MAXCOL = (NCOLBASE + NCOLSHIFT) / 2

    if (rowshift <= -np.floor(MAXROW) or rowshift >= np.ceil(MAXROW) or
            colshift <= -np.floor(MAXCOL) or colshift >= np.ceil(MAXCOL)):
        return ([np.NaN]*8)  # Nur NA zurückgeben

    bounds = [
        max(0, - (rowshift + np.floor(ROWDIFF))),
        min(NROWSHIFT, NROWSHIFT - (rowshift - np.ceil(ROWDIFF))),
        max(0, - (colshift + np.floor(COLDIFF))),
        min(NCOLSHIFT, NCOLSHIFT - (colshift - np.ceil(COLDIFF))),
        max(0, (rowshift + np.floor(ROWDIFF))),
        min(NROWBASE, NROWBASE + (rowshift - np.ceil(ROWDIFF))),
        max(0, (colshift + np.floor(COLDIFF))),
        min(NCOLBASE, NCOLBASE + (colshift - np.ceil(COLDIFF)))]
    return bounds
