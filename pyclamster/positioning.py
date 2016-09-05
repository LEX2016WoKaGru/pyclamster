# -*- coding: utf-8 -*-
"""
Created on 03.09.16

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

# Internal modules
from . import coordinates


logger = logging.getLogger(__name__)

__version__ = "0.1"


def doppelanschnitt(azi1,azi2,ele1,ele2,pos1,pos2):
    """
    calculate the 3d position via "doppelanschnitt"

    args:
        azi1, azi2, ele1, ele2 (float): SINGLE values of azimuth and elevation
            of different devices at positions 1 and 2 (radians)
        pos1, pos2 (np.array) [x,y,z] of different devices at positions
            1 and 2 (metres)

    returns:
        
    """

    e1 = np.array([np.sin(azi1)*np.cos(ele1),
                   np.cos(azi1)*np.cos(ele1),
                   np.sin(ele1)])

    e2 = np.array([np.sin(azi2)*np.cos(ele2),
                   np.cos(azi2)*np.cos(ele2),
                   np.sin(ele2)])

    n = np.cross(e1,e2,axis=0)
    n = n/np.linalg.norm(n)

    a_s = np.array([e1,e2,n]).T
    b_s = (np.array(pos1)-np.array(pos2)).T
    a,b,c = np.linalg.solve(a_s, b_s)

    logger.debug("minimum distance: {} m".format(c))

    position = np.array(pos1 - a * e1 - n * 0.5 * c)


    return position


# multiple values
def doppelanschnitt_Coordinates3d(aziele1,aziele2,pos1,pos2):
    """
    calculate 3d position based on Coordinates3d

    args:
        aziele1,aziele2 (Coordinates3d): coordinates (azimuth/elevation) of 
            devices 1 and 2. These have to be northed
        pos1, pos2 (Coordinates3d): length 1 coordinates (x,y,z) of devices
            1 and 2

    returns:
        positions (Coordinates3d): (x,y,z) positions taken from
            Doppelanschnitt.
    """
    ae1 = copy.deepcopy(aziele1)
    ae2 = copy.deepcopy(aziele2)
    logger.debug("copied aziele1:\n{}".format(ae1))
    logger.debug("copied aziele2:\n{}".format(ae2))
    # turn to north
    ae1.fill(
        azimuth   = ae1.azimuth,
        elevation = ae1.elevation,
        radius    = 1
        )
    ae1.change_parameters(
        azimuth_offset = 3/2 * np.pi,
        azimuth_clockwise = True,
        elevation_type = "ground",
        keep = {'x','y','z'}
        )

    ae2.fill(
        azimuth   = ae2.azimuth,
        elevation = ae2.elevation,
        radius = 1
        )
    ae2.change_parameters(
        azimuth_offset = 3/2 * np.pi, 
        azimuth_clockwise = True,
        elevation_type = "ground",
        keep = {'x','y','z'}
        )
    logger.debug("ae1 after turning: \n{}".format(ae1))
    logger.debug("ae2 after turning: \n{}".format(ae2))

    # convert given positions to numpy array
    position1 = np.array([pos1.x,pos1.y,pos1.z])
    logger.debug("position1: \n{}".format(position1))
    position2 = np.array([pos2.x,pos2.y,pos2.z])
    logger.debug("position2: \n{}".format(position2))

    # loop over all azimuth/elevation values
    x = [];y = [];z = [] # start with empty lists
    for azi1,azi2,ele1,ele2 in zip(
        ae1.azimuth.ravel(),   ae2.azimuth.ravel(), 
        ae1.elevation.ravel(), ae2.elevation.ravel()):
        #print(azi1.shape, azi2.shape, ele1.shape, ele2.shape)
        logger.debug("azi1: {}, azi2: {}, ele1: {}, ele2: {}".format(azi1, azi2, ele1, ele2))
        # calculate 3d doppelanschnitt position
        xnew, ynew, znew = doppelanschnitt(
            azi1=azi1,azi2=azi2,ele1=ele1,ele2=ele2,
            pos1=position1,pos2=position2)

        x.append(xnew)
        y.append(ynew)
        z.append(znew)

    # merge new coordinates
    out = coordinates.Coordinates3d(x=x,y=y,z=z,shape = ae1.shape)

    return out

class Projection(object):
    def __init__(self, zone=32):
        try:
            import pyproj
        except:
            raise ImportError('pyproj isn\'t installed yet')

        proj = '+proj=tmerc +lat_0=0 +lon_0=9 +k=1 +x_0=3500000 +y_0=0 +ellps=bessel +datum=potsdam +units=m +no_defs'
        self.p = pyproj.Proj(proj)

    def lonlat2xy(self, lon, lat, return_coordinates=False):
        """
        Method to calculate x, y coordinates from latitudes and longitudes.
        Args:
            lon (float/numpy array): Longitudes in decimal degree as array or as float.
            lat (float/numpy array): Latitudes in decimal degree as array or as float.
            return_coordinates (optional[bool]): If the output coordinates should be a Coordinates3d instance. Default is False.
        Returns:
            pos (tuple/Coordinates3d): The x, y position as tuple or as Coordinates3d instance, depending on the coordinates argument.
        """
        pos = self.p(lon, lat)
        if return_coordinates:
            pos = coordinates.Coordinates3d(
                x=pos[0],
                y=pos[1],
                azimuth_offset=3 / 2 * np.pi,
                azimuth_clockwise=True
            )
        return pos

    def xy2lonlat(self, x, y=None):
        """
        Method to calculate longitude and latitude coordinates from x and y coordinates.
        Args:
            x (Coordinates3d/float/numpy array): The x coordinate as numpy array or as float. If this is a Coordinates3d instance, the x and y coordinates are calculated out of this instance.
            y (optional[float/numpy array): The x coordinate as numpy array or as float. This argument is unnecessary if the x coordiante is a Coordinates3d instance. Default is None.
        Returns:
            pos (tuple[float/numpy array]): The longitudes and latitudes as tuple of floats or as tuple of numpy arrays if the input arguments are also a numpy array.
        """
        if isinstance(x, coordinates.Coordinates3d):
            coords = x
            x = coords.x
            y = coords.y
        pos = self.p(x, y, inverse=True)
        return pos
