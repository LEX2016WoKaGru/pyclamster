# -*- coding: utf-8 -*-
"""
Created on 18.05.16
Created for pyClamster

@author: Tobias Sebastian Finn, tobias.sebastian.finn@studium.uni-hamburg.de

    Copyright (C) {2016}  {Tobias Sebastian Finn}

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


__version__ = "0.0.0.1"


class Camera(object):
    def __init__(self):
        self.image_series = {timestamp: Image}
        self.displacement = {Azi, Ele}

    def getElevation(self):
        pass


class Image(object):
    def __init__(self, data=None):
        self.data = data
        self.degrees = {"X": 180, "Y": 180}

    def loadImage(self, path):
        pass

    def saveImage(self, path):
        pass

    def _calcCenter(self):
        pass

    def crop(self, px_x=960, px_y=960, center=True):
        center = self._calcCenter()
        self.data = self.data[center[0]-px_x:center[0]+px_x, center[1]]
        pass

    def cropRound(self, px=1920):
        pass

    def cropDegree(self, deg=45):

    def mean(self):
        pass

    def std(self):
        pass

    def getElevation(self):
        pass

    def getAzimuth(self, dis_in_deg=0):
        pass


class MaskedImage(object):
    def __init__(self, raw):
        assert isinstance(raw, Image)
        self.raw = raw
        self.masks = masks

class TrashDetector(object):
    def detect(self, image):
        return Image(image_wo_trash)


class Cloud(MaskedImage):
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



if "__name__" == "__main__":
    raw = Image()
    raw.loadImage(path)
    raw.cropDegree(60)
    cleaned = TrashDetector().detect(raw)
    =Cloud_cam1+Cloud_cam2
