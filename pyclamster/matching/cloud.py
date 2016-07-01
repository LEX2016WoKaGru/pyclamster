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
import warnings

# External modules
import numpy as np
import scipy.ndimage

# Internal modules
from .matching import ProbabilityMapping, ProbabilityMap
from ..image import Image
from ..clustering import preprocess

__version__ = ""


class BaseCloud(object):
    def __init__(self, data):
        pass


class Cloud(BaseCloud):
    def __init__(self, data, preprocess=[]):
        """
        This is a clustered cloud from one camera.
        Args:
            data (Image): The cloud data as Image instance.
                In the most cases the Image data is masked with the mask of the
                cloud.
            preprocess (list[Preprocessing layers]): List with layers to
                preprocess the cloud before the matching of different clouds.
                A layer is for example a layer, which normalize the channels,
                so that the mean of the channels is 0 and the standard
                deviation equals 1.
                Default is an empty list.
        """
        self.preprocessing = preprocess
        self.data = data
        self.__merge_cloud_type = SpatialCloud

    def _preprocess(self, data):
        output = Image(data)
        if self.preprocessing:
            for layer in self.preprocessing:
                output.data = layer.preprocess(output.data)
        return output

    def get_center(self):
        """
        Method to get the center of the cloud.
        This method is based on binary erosion to get a mass center.
        Returns:
            center (tuple[int]): The center of the cloud with (x, y) position.
        """
        erosed_array = ~self.data.data.mask[:, :, 0]
        while erosed_array.sum() != 0:
            old_array = erosed_array
            erosed_array = scipy.ndimage.binary_erosion(erosed_array)
        ind = old_array.nonzero()
        center = (int((np.max(ind[1]) + np.min(ind[1])) / 2),
                  int((np.max(ind[0]) + np.min(ind[0])) / 2))
        return center

    def match(self, clouds, mapping=None, w=None):
        """
        Method to get the matching correlation with given clouds.
        Args:
            clouds (list[Cloud]): The clouds to be matched with this cloud.
            mapping (optional[Class]): The matching algorithm.
                Default is None, so this is set in the initialization to the
                ProbabilityMapping algorithm.
            w (optional[list[float]]): Weights to weight the channels for the
                matching differently.
                Default is None, so the weights are set all to 1.

        Returns:
            prob_clouds (list[numpy array]): The probability map between this
                and the given clouds, created.
        """
        if mapping is None:
            mapping = ProbabilityMapping(w=w)
        prob_clouds = []
        for c in clouds:
            prob_map = mapping.calc_map(self.data, c.data)
            prob_clouds.append(prob_map)
        return prob_clouds

    def merge(self, cloud, mapping=None, w=None):
        """
        Method to merge cloud spatially. This method is based on a template
        matching algorithm.
        Args:
            cloud (list[Cloud]): The clouds to be matched with this cloud.
            mapping (optional[Class]): The matching algorithm.
                Default is None, so this is set in the initialization to the
                ProbabilityMapping algorithm.
            w (optional[list[float]]): Weights to weight the channels for the
                matching differently.
                Default is None, so the weights are set all to 1.

        Returns:
            matched_cloud (SpatialCloud): The matched cloud with the
                given cloud.
            prob_map (ProbabilityMap): The probability map between the two
                clouds.
        """
        if mapping is None:
            mapping = ProbabilityMapping(w=w)
        prob_map = mapping.calc_map(self.data, cloud.data)
        #matched_cloud = self.__merge_cloud_type(*prob_map.clouds, prob_map)
        return matched_cloud, prob_map


class SpatialCloud(Cloud):
    def __init__(self, cloud1, cloud2, mapping=None, preprocess=[]):
        super().__init__([cloud1, cloud2], preprocess)
        # if map is None:
        #     prob_map = self.data[0].merge(self.data[1])[1]
        # else:
        #     assert isinstance(map, ProbabilityMap),\
        #         'The map is not an available map for this version.'
        #     self.map = map
        self.position = None

    def _preprocess(self, data):
        return [super()._preprocess(c) for c in data]

    def _calc_position(self, d=100):
        """
        Method to calculate the x, y, z position of the spatial matched cloud.
        Args:
            d (float): Distance of the cameras in metres.
        Returns:
            positions (masked numpy array): The position for every matched
                pixel of the two clouds. The mask is applied where the clouds
                are disagreed. The position is relative to the first cloud.
                Shape of the array: [width of c1, height of c1, 3].
                The first channel is for the x position, the second for the y
                and the third for the cloud height.
        """
        def calc_radius(azi1, azi2):
            return d*np.sin(azi1)/np.sin(np.pi-azi1-azi2)

        def calc_x(r, azi):
            return r*np.sin(np.pi/2-azi)

        def calc_y(r, azi):
            return r*np.cos(np.pi/2-azi)
        def calc_h(Ds, ele1, ele2):
            return np.cos(ele1)*np.cos(ele2)*Ds/np.sin(ele1+ele2)
        azi = [self.data[0].data.coordinates.azimuth, self.data[1].data.coordinates.azimuth]
        ele = [self.data[0].data.coordinates.elevation, self.data[1].data.coordinates.elevation]
        R = [calc_radius(azi[0], azi[1]), calc_radius(azi[1], azi[0])]
        X = [calc_x(R[0], azi[0]), calc_x(R[1], azi[1])]
        Y = [calc_y(R[0], azi[0]), calc_y(R[1], azi[1])]
        H = calc_h(R[0]+R[1], ele[0], ele[1])
        # TODO check if the formulas are right.
        return X, Y, H


class TemporalCloud(Cloud):
    pass
