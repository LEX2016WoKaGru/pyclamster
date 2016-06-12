# -*- coding: utf-8 -*-
"""
Created on 27.05.16

Created for pyclamster

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
import pickle
import warnings

# External modules
import numpy as np
import scipy.misc
import scipy.ndimage

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


# Internal modules
from pyclamster import Image
from pyclamster.clustering.preprocess import LCN, ZCA
from pyclamster.clustering.kmeans import KMeans
from pyclamster.clustering.functions import localBrightness, rbDetection

warnings.catch_warnings()
warnings.filterwarnings('ignore')

"""
0 = Sky
1 = Clouds
2 = Trash
3 = Trash
4 = Sky
5 = Trash
6 = Clouds

"""

__version__ = "0.1"

k_cluster = 2
good_angle = 45
center = int(1920/2)
good_angle_dpi = int(np.round(1920 / 180 * good_angle))
neighbour_size = 10

denoising_ratio = 10

anomaly_images = None

for i in range(1, 5):
    image = Image(u'Image_Wkm_Aktuell_{0:d}.jpg'.format(i))

    image.data = LCN(size=(50,50,3), scale=False).fit_transform(image.data)
    #image.data = LCN(size=(30,30,1), scale=True).fit_transform(image.data)
    image.data = image.data[center - good_angle_dpi:center + good_angle_dpi,
                            center - good_angle_dpi:center + good_angle_dpi]

    anomaly_image = rbDetection(image.data)
    scipy.misc.imsave(u'test_anomaly_{0:d}.jpg'.format(i), anomaly_image)

    w, h = original_shape = tuple(anomaly_image.shape)
    anomaly_image = np.reshape(anomaly_image, (w * h, -1))
    if anomaly_images is None:
        anomaly_images = anomaly_image
    else:
        anomaly_images = np.r_[anomaly_images, anomaly_image]

kmeans = KMeans(2).fit(anomaly_images)
anomaly_labels = kmeans.labels
anomaly_labels = anomaly_labels.splitUp(indices_or_sections=4)

for label in anomaly_labels:
    label.reshape((w, h), replace=True)
    scipy.misc.imsave(u'anomaly_labels_{0:d}.png'.format(i + 1),
                      label.labels)
