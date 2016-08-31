# -*- coding: utf-8 -*-
"""
Created on 31.08.16

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
import re

# External modules
import simplekml
import numpy as np

# Internal modules

def cloud2kml(cloud, file_path=None):
    """
    Function to plot a cloud array into a kml file.
    Args:
        cloud (numpy array): The cloud informations the shape should be (nr_x, nr_y, 7).
            The seven channels are: longitude, latitude, altitude, red channel,
            green channel, blue channel, alpha channel.

        file_path (str): The file path where the kml file should be saved.
    """
    if isinstance(file_path, simplekml.kml.Kml):
        kml_file = file_path
    else:
        kml_file = simplekml.Kml()
    print(type(kml_file))
    multipnt = kml_file.newmultigeometry(name='Cloud 0')
    for (x, y), value in np.ndenumerate(cloud[:,:,0]):
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


base_height = 1000

kml_file = simplekml.Kml()

input = np.empty((10,10,7))
input[:,:,0] = np.linspace(11, 11.2, 10).reshape((1,10))
input[:,:,1] = np.linspace(54.4, 54.6, 10).reshape((10,1))
input[:,:,2] = base_height+np.random.normal(0,100,(10,10))
input[:,:,3] = np.random.normal(128,50,(10,10))
input[:,:,4] = np.random.normal(65,10,(10,10))
input[:,:,5] = np.random.normal(70,10,(10,10))
input[:,:,6] = np.random.normal(254,0.001,(10,10))
cloud2kml(input, kml_file)

input = np.empty((10,10,7))
input[:,:,0] = np.linspace(11.3, 11.45, 10).reshape((1,10))
input[:,:,1] = np.linspace(54.8, 54.9, 10).reshape((10,1))
input[:,:,2] = base_height+500+np.random.normal(0,100,(10,10))
input[:,:,3] = np.random.normal(128,50,(10,10))
input[:,:,4] = np.random.normal(140,10,(10,10))
input[:,:,5] = np.random.normal(110,20,(10,10))
input[:,:,6] = np.random.normal(128,50,(10,10))
cloud2kml(input, kml_file)

kml_file.save('/home/tfinn/Desktop/test.kml')
