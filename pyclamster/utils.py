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

def calc_color(val,vmin,vmax):
    if val < vmin:
        val = vmin
    if val > vmax: 
        val = vmax
    c = (val-vmin)/(vmax-vmin)
    r,g,b = np.array([1-c,0,c])*99
    r = str(int(r))
    g = str(int(g))
    b = str(int(b))
    if len(r) == 1: r = '0'+r;
    if len(g) == 1: g = '0'+g;
    if len(b) == 1: b = '0'+b;
    return '#'+r+g+b


