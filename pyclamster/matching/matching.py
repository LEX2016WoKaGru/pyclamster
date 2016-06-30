# -*- coding: utf-8 -*-
"""
Created on 27.06.16

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
import numpy as np

#from skimage

# Internal modules
#from pyclamster.matching.cloud import SpatialCloud, TemporalCloud

__version__ = "0.1"


class Matching(object):
    def __init__(self, w=None):
        self.w = w

    def matching(self, clouds1, clouds2, temporal=False):
        maps = [[], []]
        best_maps = []
        for k, c in enumerate(clouds1):
            maps[0, k] = c.merge(clouds2)[2]
        for k, c in enumerate(clouds2):
            maps[1, k] = c.merge(clouds1)[2]
        # TODO Algorithm for best maps needs to be implemented
        best_clouds = None
        return best_clouds



class ProbabilityMapping(object):
    def __init__(self, w=None):
        self.w = w

    def _normalize_weights(weights):
        normalized_weights = [w/np.sum(weights) for w in weights]
        return normalized_weights

    def _check_dimensions(self, data):
        if data.shape[-1] != len(self.w):
            raise ValueError('The dimension of the weights and of the cloud'
                             'channels are not the same!')

    def calc_map(self, cloud1, cloud2):
        if self.w is None:
            self.w = [1] * cloud1.data.shape[2]
        self.w = self._normalize_weights(self.w)
        self._check_dimensions(cloud1)
        self._check_dimensions(cloud2)
        return ProbabilityMap(cloud1, cloud2, self.w)


class ProbabilityMap(object):
    def __init__(self, cloud1, cloud2, w):
        self.clouds = [cloud1, cloud2]
        self.w = w
        self.map = self._calc_map()

    def __call__(self):
        return self.map

    def _calc_map(self):
        """
        Method to get the probability map of two different clouds.
        Args:
            cloud1 (Cloud/SpatialCloud): The first cloud.
            cloud2 (Cloud/SpatialCloud): The cloud in which the first cloud
                should be moved.
            w (list[float]): Weights for the different channels.

        Returns:
            probability_map (numpy array): The probability map for every
                possible matching point. Should have the same data shape like
                the second cloud.
        """
        probability_map = None
        """
        TODO
        template = cloud_store.cutMask(cutted_image, [1,])
        result = match_template(cutted_image.data, template.data, pad_input=True, mode='reflect', constant_values=0)
        Second cloud cut has to have the same size as the first cloud cut
        """
        return probability_map

    def get_best(self):
        """
        Method to get the best point of the map.
        Returns:
            best (dict[float]): A dict with information about the best point
                within the map.
        """
        # TODO here is a method to get the best point needed
        best = {'prob': None, 'x': None, 'y': None}
        return best
