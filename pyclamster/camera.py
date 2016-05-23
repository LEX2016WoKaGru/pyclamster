# -*- coding: utf-8 -*-
"""
Created on 23.05.16

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

# External modules

# Internal modules
from .image import Image


__version__ = "0.1"


class Camera(object):
    """
    class that holds a series of images
    and camera properties
    """
    def __init__(self):
        self.image_series     = {timestamp: Image}
        self.Azi_displacement = {timestamp: Azi}
        self.Ele_displacement = {timestamp: Ele}

    def getImage(self,timestamp):
        """
        return an instance of class Image() for the timestamp timestamp
        """
        image = Image(timestamp) # TODO
        return image
