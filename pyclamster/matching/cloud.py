# -*- coding: utf-8 -*-
"""
Created on 10.06.16

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
from ..image import Image


__version__ = ""

class BaseCloud(Image):
    def __init__(self, image):
        pass


class Cloud(Image):
    def __init__(self, image, mask, w=None):
        self.data = image
        self.label = mask
        if w is None:
            w = [1] * self.data.shape[2]
        self.matching_algorithm = Matching(w=w)
        self.data = Preprocess(data)

    def merge(self, clouds):
        """
        Method to merge colud spatially. This method is based on a template
        matching algorithm.
        Args:
            clouds (list[Cloud]):
        Returns:
            matched_cloud (SpatialCloud):
            prob_clouds (list[numpy array])
            informations (dict[informations]):
        """
        prob = []
        for c in clouds:
            c_data = Preprocess(data)
            prob.append(self.matching_algorithm.match(self.data, c.data))
        pass


class SpatialCloud(Cloud):
    pass


class TemporalCloud(BaseCloud):
    pass
