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

# Internal modules


__version__ = "0.1"



class Matching(object):
    def __init__(self, w):
        self.w = self._normalize_weights(w)
        self.mapping = ProbabilityMap(self.w)

    def _calc_prob(self, cloud1, cloud2):
        return self.mapping.probability(cloud1, cloud2)

    def _normalize_weights(self, weights):
        normalized_weights = [w/np.sum(weights) for w in weights]
        return normalized_weights

    def match(self, cloud1, cloud2):
        """
        Matches two clouds and get the probability distribution map and the
        best match for the two clouds.
        Args:
            cloud1 (Cloud/SpatialCloud): The first cloud.
            cloud2 (Cloud/SpatialCloud): The cloud, in which the first cloud
                should be moved.

        Returns:
            best_match (dict[prob, x, y]): The best matching point in the
                second cloud.
                'prob': the probability that the clouds are belongs together
                    in this specific point.
                'x': The x position of the best matching point.
                'y': The y position of the best matching point.
            probability_map (numpy array): The probability map for every
                possible matching point. Should have the same data shape like
                the second cloud.
        """
        probability_map = self._calc_prob(cloud1, cloud2)
        # here is missing a method to get the best point!
        best_match = {'prob': None, 'x': None, 'y': None}
        return best_match, probability_map


class ProbabilityMap(object):
    def __init__(self, w):
        self.w = w

    def _check_dimensions(self, data):
        if data.shape[-1] != len(self.w):
            raise ValueError('The dimension of the weights and of the cloud'
                             'channels are not the same!')

    def get_probability(self, cloud1, cloud2):
        """
        Method to get the probability map of two different clouds.
        Args:
            cloud1 (Cloud/SpatialCloud): The first cloud.
            cloud2 (Cloud/SpatialCloud): The cloud in which the first cloud
                should be moved.

        Returns:
            probability_map (numpy array): The probability map for every
                possible matching point. Should have the same data shape like
                the second cloud.
        """
        probability_map = None
        # Needs algorithm to create the probability map
        return probability_map