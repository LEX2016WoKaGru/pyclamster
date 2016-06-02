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

# External modules
import numpy as np
import scipy.misc
import scipy.ndimage

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# from pyextremelm.builder import ExtremeLearningMachine
# from pyextremelm.metrics import NoMetric
# from pyextremelm.builder.layers.unsupervised import ELMAE

# Internal modules


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
    image = scipy.ndimage.imread(u'Image_Wkm_Aktuell_{0:d}.jpg'.format(i),
                                 mode="RGB")
    scipy.misc.imsave(
        u'test_original_{0:d}.jpg'.format(i),
        image[center - good_angle_dpi:center + good_angle_dpi,
              center - good_angle_dpi:center + good_angle_dpi, :])

    image_mean = scipy.ndimage.filters.uniform_filter(
        image, size=(neighbour_size, neighbour_size, 3), mode="constant")
    image = image-image_mean
    anomaly_image = (image[:,:,2]-image[:,:,0])/(image[:,:,2]+image[:,:,0])
    anomaly_image[(image[:,:,2]+image[:,:,0])==0] = 0

    anomaly_image = anomaly_image[center - good_angle_dpi:center + good_angle_dpi,
                                  center - good_angle_dpi:center + good_angle_dpi]
    scipy.misc.imsave(u'test_anomaly_{0:d}.jpg'.format(i), anomaly_image)

    w, h = original_shape = tuple(anomaly_image.shape)
    anomaly_image = np.reshape(anomaly_image, (w * h, -1))
    if anomaly_images is None:
        anomaly_images = anomaly_image
    else:
        anomaly_images = np.r_[anomaly_images, anomaly_image]

print(anomaly_images.shape)
kmeans = MiniBatchKMeans(n_clusters=k_cluster, random_state=0).fit(anomaly_images)
anomaly_labels = np.abs((-1)*kmeans.labels_)

for i in range(0, 4):
    anomaly_single_label = anomaly_labels[i * (w * h):(i + 1) * (w * h)]
    anomaly_single_label = np.reshape(anomaly_single_label, (w, h))
    scipy.misc.imsave(u'anomaly_labels_{0:d}.jpg'.format(i + 1),
                      anomaly_single_label)
