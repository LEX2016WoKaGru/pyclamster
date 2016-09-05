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
import scipy.misc
import scipy.ndimage

# Internal modules
from .matching import ProbabilityMap
from ..positioning import Projection
from ..utils import shift_matrix, cloud2kml
from ..image import Image
from ..clustering import preprocess
from .. import positioning

__version__ = ""


class BaseCloud(object):
    def __init__(self):
        pass


class Cloud(BaseCloud):
    def __init__(self, image, label, preprocess=[]):
        """
        This is a clustered cloud from one camera.
        Args:
            data (Image): The cloud data as Image instance.
                In the most cases the Image data is masked with the mask of the
                cloud.
            label (numpy array): The numpy array labels, where True are the
                cloud pixels and False are the non cloud pixels.
            preprocess (list[Preprocessing layers]): List with layers to
                preprocess the cloud before the matching of different clouds.
                A layer is for example a layer, which normalize the channels,
                so that the mean of the channels is 0 and the standard
                deviation equals 1.
                Default is an empty list.
        """
        self.preprocessing = preprocess
        self.image = image
        self.label = label
        self.data = self.image.data
        self.__merge_cloud_type = SpatialCloud
        super().__init__()

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
        erosed_array = ~self.image.data.mask[:, :, 0]
        while erosed_array.sum() != 0:
            old_array = erosed_array
            erosed_array = scipy.ndimage.binary_erosion(erosed_array)
        ind = old_array.nonzero()
        center = (int((np.max(ind[1]) + np.min(ind[1])) / 2),
                  int((np.max(ind[0]) + np.min(ind[0])) / 2))
        return center

    def merge(self, clouds, w=None):
        """
        Method to merge cloud spatially. This method is based on a template
        matching algorithm.
        Args:
            clouds (list[Cloud]): The clouds to be matched with this cloud.
            w (optional[list[float]]): Weights to weight the channels for the
                matching differently.
                Default is None, so the weights are set all to 1.

        Returns:
            merged_result (list[list]): list of [prob_map,matched_cloud] for every given cloud
            matched_cloud (SpatialCloud): The matched cloud with the
                given cloud.
            prob_map (ProbabilityMap): The probability map between the two
                clouds.
        """
        if not isinstance(clouds, list):
            clouds = [clouds]
        merged_result = []
        for c in clouds:
            prob_map = ProbabilityMap(self, c, w)
            matched_cloud = SpatialCloud(cloud1=self, cloud2=c,
                                         prob_map=prob_map)
            merged_result.append([prob_map, matched_cloud])
        return merged_result


class SpatialCloud(Cloud):
    def __init__(self, cloud1, cloud2, prob_map):
        # super().__init__([cloud1, cloud2], preprocess)
        # if map is None:
        #     prob_map = self.data[0].merge(self.data[1])[1]
        # else:
        #     assert isinstance(map, ProbabilityMap),\
        #         'The map is not an available map for this version.'
        #     self.map = map
        self.clouds = [cloud1, cloud2]
        self.prob_map = prob_map
        self.positions = None

    def calc_overlapping(self):
        """
        Method to get information of the overlapping array slices
        Args:
            cloud1:
            cloud2:
            prop_map:

        Returns:

        """
        c1_data = self.clouds[0].image.data
        c2_data = self.clouds[1].image.data
        best_point = self.prob_map.get_best()
        shift = [best_point['point'][0] - int(c1_data.shape[0] / 2),
                 best_point['point'][1] - int(c1_data.shape[1] / 2)]
        bounds = shift_matrix(c1_data.shape, c2_data.shape, shift[0], shift[1])
        self.clouds[0].image = self.clouds[0].image.cut(
            [bounds[6], bounds[4], bounds[7], bounds[5]])
        self.clouds[0].label = self.clouds[0].label[bounds[4]:bounds[5],
                               bounds[6]: bounds[7]]
        self.clouds[1].image = self.clouds[1].image.cut(
            [bounds[2], bounds[0], bounds[3], bounds[1]])
        self.clouds[1].label = self.clouds[1].label[bounds[0]:bounds[1],
                               bounds[2]: bounds[3]]
        return self

        # def _preprocess(self, data):
        #     return [super()._preprocess(c) for c in data]

    def write2kml(self, kml_path):
        """
        Method to write the cloud data to a kml file.
        file_path (str/simplekml.kml.Kml): The file path where the kml file
            should be saved. If file_path is an instance of simplekml.kml.Kml
            the cloud is written to this instance.

        Returns:
            kml_file (simplekml.kml.Kml): If a kml file was the argument, this method returns this kml_file.
        """
        c1_data = self.clouds[0].image.data
        c2_data = self.clouds[1].image.data
        c1_label = self.clouds[0].label
        c2_label = self.clouds[1].label
        lat, lon = self.latlon
        pos = np.expand_dims(np.empty_like(c1_label), axis=2)
        pos = np.append(pos, lat[..., np.newaxis], axis=2)
        pos = np.append(pos, lon[..., np.newaxis], axis=2)
        pos = np.append(pos, self.positions.z[..., np.newaxis], axis=2)
        colors = c1_data
        colors[np.logical_and(c1_label, c2_label)] = (c1_data[np.logical_and(
            c1_label, c2_label)] + c2_data[np.logical_and(c1_label,
                                                          c2_label)]) / 2
        colors[np.logical_and(~c1_label, c2_label)] = c2_data[
            np.logical_and(~c1_label, c2_label)]
        colors = colors/256
        colors[np.logical_and(~c1_label, ~c2_label)] = np.NaN
        pos = np.append(pos, colors, axis=2)
        alpha = (c1_label + c2_label) / 2
        pos = np.append(pos, alpha[..., np.newaxis], axis=2)
        kml_file = cloud2kml(pos, self.clouds[0].image.time, kml_path)
        if not isinstance(kml_path, str):
            return kml_file

    @property
    def height(self):
        return self.get_height()

    @property
    def latlon(self):
        if not self.positions is None:
            lat, lon = Projection().xy2lonlat(self.positions)
            return lat, lon
        else:
            print('The positions aren\'t calculated yet!')

    def get_height(self):
        if not self.positions is None:
            return np.mean(self.positions.coordinates.z)
        else:
            print('The positions aren\'t calculated yet!')

    def calc_position(self):
        """
        Method to calculate the x, y, z position of the spatial matched cloud.
        """
        self.positions = positioning.doppelanschnitt_Coordinates3d(
            aziele1=self.clouds[0].image.coordinates,
            aziele2=self.clouds[1].image.coordinates,
            pos1=self.clouds[0].image.position,
            pos2=self.clouds[1].image.position,
        )


class TemporalCloud(Cloud):
    pass
