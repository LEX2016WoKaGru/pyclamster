# -*- coding: utf-8 -*-
"""
Created on 18.05.16
Created for pyClamster

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
import os,sys,time

# External modules
import scipy.ndimage
import scipy.misc
import scipy as sp
import numpy as np

# Internal modules


__version__ = "0.0.0.1"






        

class TrashDetector(object):
    def detect(self, image):
        return Image(image_wo_trash)


class Cloud(Image):
    def __add__(self, other):
        pass


class CloudMatcher(object):
    def match(self, Cloud1, Cloud2):
        pass

    def match_bulk(self, clouds=[]):
        pass

class WettermastData(object):
    pass


class Ceilometer(object):
    pass


if __name__ == "__main__":
    raw = Image()
    raw.loadImage("/home/tfinn/Projects/pyclamster/pyclamster/k-means/test/cloudy.jpg")
    raw.saveImage("2.jpg")
    #=Cloud_cam1+Cloud_cam2
