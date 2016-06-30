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
from .matching import Matching
from ..image import Image
from ..clustering import preprocess


__version__ = ""

class BaseCloud(Image):
    def __init__(self, data):
        pass


class Cloud(BaseCloud):
    def __init__(self, data, preprocess=[], matching=None, w=None):
        if w is None:
            w = [1] * self.data.shape[2]
        self.preprocessing = preprocess
        if matching is None:
            self.matching_algorithm = Matching(w=w)
        else:
            self.matching_algorithm = matching
        self.data = self._preprocess(data)

    def _preprocess(self, data):
        output = data
        if self.preprocessing:
            for layer in self.preprocessing:
                output = layer.preprocess(data)
        return output

    def merge(self, clouds, threshold):
        """
        Method to merge colud spatially. This method is based on a template
        matching algorithm.
        Args:
            clouds (list[Cloud]):
            thres (float): Level of probability,
        Returns:
            matched_cloud (SpatialCloud):
            prob_clouds (list[numpy array])
            informations (dict[informations]):
        """
        prob = []
        for c in clouds:
            best, prob_map = self.matching_algorithm.match(self.data, c.data)
            best
        pass


class SpatialCloud(Cloud):
    def __init__(self, c1, c2, preprocess=[], matching=None, w=None):
        data
        super().__init__([c1, c2], preprocess, matching, w)

    def _preprocess(self, data):
        return [super()._preprocess(c) for c in data]



class TemporalCloud(BaseCloud):
    pass
