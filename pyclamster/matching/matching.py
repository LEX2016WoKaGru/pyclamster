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
import logging


# External modules
import numpy as np
from skimage.feature import match_template

# from skimage

# Internal modules
# from pyclamster.matching.cloud import SpatialCloud, TemporalCloud

__version__ = "0.1"

logger = logging.getLogger(__name__)



class Matching(object):
    """
    object used to handel the matching of cloud objects
    Args:
        w(list[float]): weights used for the given channels (max value 1, dim = cloud.data.shape[2])
    """

    def __init__(self, w=None, greyscale=False):
        self.w = w
        self.greyscale = greyscale

    def matching(self, clouds1, clouds2, min_match_prob=0.75):
        """
        matching to lists of clouds together by creating a ProbabilityMap to compare clouds
        Args:d
            clouds1 (list[Cloud/SpatialCloud]): list of clouds from first camera
            clouds2 (list[Cloud/SPatialCloud]): list of clouds from second camera
            min_match_prob (float): used to determine minimal matching probability to accept
        Returns:
            matched_clouds ():
            matched_idx (list[list[int,int]]): list of matched cloud indicees (clouds1_idx,clouds2_idx)
        """
        mergedC = [[], []]
        best_probs = [[], []]
        best_dicts = [[], []]
        best_match = [[], []]
        matched_clouds = []
        matched_idx = [[-1, -1]]
        for idx_c1, c in enumerate(clouds1):
            # merge one cloud1 with all clouds2(= PropabilityMap)
            mergedC[0].append(c.merge(clouds2, self.w, self.greyscale))
            # get best matching points out of matches(= dict)
            best_dicts[0].append(
                [mergedC[0][idx_c1][idx_c2][0].get_best() for idx_c2 in
                 range(len(clouds2))])
            # get probabilities out of dicts (= float)
            best_probs[0].append(
                [best_dicts[0][idx_c1][idx_c2]['prob'] for idx_c2 in
                 range(len(clouds2))])
            # get cloud2 idx for highest percentage (= int)
            bm = np.argmax(best_probs[0][idx_c1])
            # store matched cloud1 -> cloud2 if probability higher than 'min_match_prob'
            # print(str(best_dicts[0][idx_c1][bm]['prob'])+' > '+str(min_match_prob)+' and not '+str([idx_c1, bm])+' in '+str(matched_idx))
            if best_dicts[0][idx_c1][bm]['prob'] > min_match_prob and not [idx_c1, bm] in matched_idx:
                matched_clouds.append(mergedC[0][idx_c1][bm])
                matched_idx.append(
                    [idx_c1, bm])  # care that there will be no double match

        for idx_c2, c in enumerate(clouds2):
            mergedC[1].append(c.merge(clouds1, self.w, self.greyscale))
            best_dicts[1].append(
                [mergedC[1][idx_c2][idx_c1][0].get_best() for idx_c1 in
                 range(len(clouds1))])
            best_probs[1].append(
                [best_dicts[1][idx_c2][idx_c1]['prob'] for idx_c1 in
                 range(len(clouds1))])
            bm = np.argmax(best_probs[1][idx_c2])
            if best_dicts[1][idx_c2][bm]['prob'] > min_match_prob and not [bm, idx_c2] in matched_idx:
                matched_clouds.append(mergedC[1][idx_c2][bm])
                matched_idx.append(
                    [bm, idx_c2])  # care that there will be no double match

        return matched_clouds, matched_idx  # returns [ProbabilityMap, SpatialCloud]


class ProbabilityMap(object):
    def __init__(self, cloud1, cloud2, w, greyscale=False, template_size=0.9):
        self.greyscale = greyscale
        if len(cloud1.data.shape)==2:
            cloud1.data = cloud1.data[:,:,np.newaxis]
        if len(cloud2.data.shape)==2:
            cloud2.data = cloud2.data[:,:,np.newaxis]
        try:
            if cloud1.data.shape[2] == cloud2.data.shape[2]:
                self.clouds = [cloud1, cloud2]
            else:
                raise ValueError("error matching.PropabilityMap: cloud-dimension missmatch!")

            self.w = [1] * cloud1.data.shape[2]
            if isinstance(w, list):
                if len(w) == cloud1.data.shape[2]:
                    self.w = w
            self.w = self._normalize_weights(self.w)
            self.template_size = template_size

            self.prob_map = self._calc_map()
        except Exception as e:
            logger.info('Couldn\'t compare the two clouds, due to {0:s}'.format(e))
            self.w = w
            self.prob_map = np.array([-99999])[:, np.newaxis]

    def __call__(self):
        return self.map

    @staticmethod
    def _normalize_weights(weights):
        normalized_weights = [w / np.sum(weights) for w in weights]
        return normalized_weights

    def _calc_map(self):
        """
        Method to get the probability map of two different clouds.
        Args:
            cloud1 (Cloud/SpatialCloud): The cloud in which the first cloud should be moved.
            cloud2 (Cloud/SpatialCloud): The first cloud.
            w (list[float]): Weights for the different channels.

        Returns:
            probability_map (numpy array): The probability map for every
                possible matching point. Should have the same data shape like
                the first cloud.
        """
        main_img = np.array(
            self.clouds[0].data)  # if mask array the mask will be dismissed
        template = np.array(self.clouds[1].data)

        # make sure that the main_img is the bigger-cloud
        if main_img.shape[0] * self.template_size < template.shape[0]:
            margins = self._calc_max_boundary(main_img.shape[0],
                                              template.shape[0],
                                              self.template_size)
            template = template[margins[0]:margins[1], :]
        if main_img.shape[1] * self.template_size < template.shape[1]:
            margins = self._calc_max_boundary(main_img.shape[1],
                                              template.shape[1],
                                              self.template_size)
            template = template[:, margins[0]:margins[1]]

        ### useing this with weights to do every channel on it's own
        if 0 in main_img.shape or 0 in template.shape or template.shape[0]<30 or template.shape[1]<30:
            return np.array([-99999])[:, np.newaxis]
        else:
            if self.greyscale:
                main_img = np.mean(main_img, axis=2)
                template = np.mean(template, axis=2)
                try:
                    return match_template(main_img[:, :], template[:, :],
                                          pad_input=True, mode='reflect',
                                          constant_values=0)
                except Exception as e:
                    print(main_img.shape, template.shape, e)

            else:
                return match_template(main_img, template,
                                       pad_input=True, mode='reflect',
                                       constant_values=0)


    def get_best(self):
        """
        Method to get the best point of the map.
        Returns:
            best (dict[float]): A dict with information about the best point
                within the map.
        """
        idx = self.prob_map.argmax()
        xy = np.unravel_index(idx, self.prob_map.shape)
        best = {'prob': self.prob_map[xy[0], xy[1]], 'point': xy}
        return best

    def _calc_max_boundary(self, main, temp, fact):
        """
        Method to calculate the maximal possible boundary size of template.
        Args:
            main (int): size of main image
            temp (int): size of template
            fact (float): factor to determine maximal template size as percentage (max 1) 
        Returns:
            margins (list[int]): lower boundary and upper boundary (dim = 2)
        """
        max_temp = main * fact
        additional_size = temp - max_temp
        if additional_size > 0:
            margins = [int(additional_size * .5)+1,
                       int(temp - additional_size * .5)]
        else:
            margins = [0, temp]
        if margins[0] < 0 or margins[1] > main:
            margins = [0,main]
        
        return margins
