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
from . import coordinates

__version__ = "0.1"

# create logger
logger = logging.getLogger(__name__)


# recursive list flattening
def flatten(x):
    res = []  # start with an empty list
    for v in x:  # loop over all given list elements
        if type(v) in (tuple, list):  # if element is list-like
            res.extend(flatten(v))  # extend resulting list
        else:  # if element is not list-like
            res.append(v)  # append the element
    return res  # return resulting list


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
    return x / (2 * np.pi) * 360


def pos_rad(x):
    """
    convert the given radian angle to a positive value
    args:
        x(numeric): angle in radians
    return:
        radianangle = numeric
    """
    return (x % (2 * np.pi) + 2 * np.pi) % (2 * np.pi)


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
        return ([np.NaN] * 8)  # Nur NA zurückgeben

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


def cloud2kml(cloud, date, file_path=None):
    """
    Function to plot a cloud array into a kml file. Needs simplekml!
    Args:
        cloud (numpy array): The cloud informations the shape should be (nr_x, nr_y, 7).
            The seven channels are: longitude, latitude, absolute altitude, red channel,
            green channel, blue channel, alpha channel.

        date (datetime.datetime): The date information for this cloud.

        file_path (str/simplekml.kml.Kml): The file path where the kml file
            should be saved. If file_path is an instance of simplekml.kml.Kml
            the cloud is written to this instance.
    """
    import simplekml
    if isinstance(file_path, simplekml.kml.Kml):
        kml_file = file_path
    else:
        kml_file = simplekml.Kml()
    multipnt = kml_file.newmultigeometry(name='Cloud 0')
    multipnt.timestamp.when = date.isoformat()
    for (x, y), value in np.ndenumerate(cloud[:,:,5]):
        if value != np.NaN and x<cloud.shape[0]-1 and y<cloud.shape[1]-1:
            pol = multipnt.newpolygon(name='segment {0} {1}'.format(
                cloud[x,y,0], cloud[x,y,1]), altitudemode='absolute')
            pol.outerboundaryis = [
                (cloud[x,y,0], cloud[x,y,1], cloud[x,y,2]),
                (cloud[x,y+1,0], cloud[x,y+1,1], cloud[x,y+1,2]),
                (cloud[x+1,y+1,0], cloud[x+1,y+1,1], cloud[x+1,y+1,2]),
                (cloud[x+1,y,0], cloud[x+1,y,1], cloud[x+1,y,2])]
            pol.style.linestyle.width = 0
            rgb = []
            for i in range(3,7):
                rgb.append(cloud[x,y,i].astype(int))
            pol.style.polystyle.color = simplekml.Color.rgb(*rgb)
    if isinstance(file_path, str):
        kml_file.save(file_path)
    elif not isinstance(file_path, simplekml.kml.Kml):
        return kml_file



class Projection(object):
    def __init__(self, zone=32):
        try:
            import pyproj
        except:
            raise ImportError('pyproj isn\'t installed yet')

        # set the projection to utm within zone 32 (Hamburg, Fehmarn) and a
        # WGS84 ellipsoid
        self.p = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')

    def lonlat2xy(self, lon, lat, coordinates=False):
        """
        Method to calculate x, y coordinates from latitudes and longitudes.
        Args:
            lon (float/numpy array): Longitudes in decimal degree as array or as float.
            lat (float/numpy array): Latitudes in decimal degree as array or as float.
            coordinates (optional[bool]): If the output coordinates should be a Coordinates3d instance. Default is False.
        Returns:
            pos (tuple/Coordinates3d): The x, y position as tuple or as Coordinates3d instance, depending on the coordinates argument.
        """
        pos = self.p(lon, lat)
        if coordinates:
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
        if isinstance(x, pyclamster.coordinates.Coordinates3d):
            coords = x
            x = coords.x
            y = coords.y
        pos = self.p(x, y, inverse=True)
        return pos
