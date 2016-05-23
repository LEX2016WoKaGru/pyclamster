# -*- coding: utf-8 -*-
"""
Created on 20.05.16

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
good_angle_dpi = int(np.round(1920 / 180 * good_angle))
neighbour_size = 10

denoising_ratio = 6

anomaly_images = None

for i in range(1, 5):
    image = scipy.ndimage.imread(u'Image_Wkm_Aktuell_{0:d}.jpg'.format(i),
                                 mode="RGB")
    scipy.misc.imsave(
        u'test_original_{0:d}.jpg'.format(i),
        image[1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi,
        1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi, :])

    image_mean = scipy.ndimage.filters.uniform_filter(
        image, size=(neighbour_size, neighbour_size, 3), mode="constant")

    anomaly_image = image - image_mean

    anomaly_image = anomaly_image[
                    1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi,
                    1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi, :]
    scipy.misc.imsave(u'test_anomaly_{0:d}.jpg'.format(i), anomaly_image)

    w, h, d = original_shape = tuple(anomaly_image.shape)
    anomaly_image = np.reshape(anomaly_image, (w * h, d))
    if anomaly_images is None:
        anomaly_images = anomaly_image
    else:
        anomaly_images = np.r_[anomaly_images, anomaly_image]

kmeans = MiniBatchKMeans(n_clusters=k_cluster, random_state=0).fit(
    anomaly_images)
anomaly_labels = kmeans.labels_

targets = None
mini_images = None

for i in range(0, 4):
    anomaly_single_label = anomaly_labels[i * (w * h):(i + 1) * (w * h)]
    anomaly_single_label = np.reshape(anomaly_single_label, (w, h))
    # erosion_label = scipy.ndimage.binary_erosion(
    #     anomaly_single_label,
    #     structure=np.ones((6, 6)))
    # propagation_label = scipy.ndimage.binary_propagation(
    #      erosion_label, structure=np.ones((18, 18)), mask=anomaly_single_label)
    # anomaly_single_label = scipy.ndimage.binary_opening(
    #     anomaly_single_label,
    #     structure=np.ones((denoising_ratio, denoising_ratio)))
    # anomaly_single_label = scipy.ndimage.binary_closing(
    #     anomaly_single_label,
    #     structure=np.ones((denoising_ratio, denoising_ratio)))
    # propagation_label = scipy.ndimage.binary_erosion(
    #     propagation_label,
    #     structure=np.ones((6, 6)))
    # anomaly_single_label = scipy.ndimage.binary_dilation(
    #     anomaly_single_label,
    #     structure=np.ones((6, 6)))
    scipy.misc.imsave(u'anomaly_labels_{0:d}.jpg'.format(i + 1),
                      anomaly_single_label)
    #
    # anomaly_single_label[anomaly_single_label == 2] = 1
    # anomaly_single_label[anomaly_single_label == 3] = 1
    # anomaly_single_label[anomaly_single_label == 4] = 0
    # anomaly_single_label[anomaly_single_label == 5] = 1
    # anomaly_single_label[anomaly_single_label == 6] = 1
    #
    # anomaly_single_label[anomaly_single_label == 0] = -1
    # scipy.misc.imsave(u'anomaly_labels_after{0:d}.jpg'.format(i + 1),
    #                   anomaly_single_label)
    #
    # image = scipy.ndimage.imread(u'Image_Wkm_Aktuell_{0:d}.jpg'.format(i + 1),
    #                              mode="RGB")
    #
    # for i in np.asarray(np.where(anomaly_single_label == 1)).T.tolist():
    #     if targets is None:
    #         targets = np.array([1])
    #     else:
    #         targets = np.r_[targets, np.array([1])]
    #
    #     mini_image = image[
    #                  1920 / 4 + i[0] - neighbour_size:
    #                  1920 / 4 + i[0] + neighbour_size+1,
    #                  1920 / 4 + i[1] - neighbour_size:
    #                  1920 / 4 + i[1] + neighbour_size+1]
    #     mini_image = mini_image.reshape(
    #         1, neighbour_size*2+1, neighbour_size*2+1, 3)
    #
    #     if mini_images is None:
    #         mini_images = mini_image
    #     else:
    #         mini_images = np.r_[mini_images, mini_image]
    #
    #     if mini_images.shape[0]%10000==0:
    #         scipy.misc.imshow(mini_image.reshape(neighbour_size*2+1, neighbour_size*2+1, 3))
    #
    # for i in np.asarray(np.where(anomaly_single_label == -1)).T.tolist():
    #     if targets is None:
    #         targets = np.array([1])
    #     else:
    #         targets = np.r_[targets, np.array([1])]
    #
    #     mini_image = image[
    #                  1920 / 4 + i[0] - neighbour_size:
    #                  1920 / 4 + i[0] + neighbour_size+1,
    #                  1920 / 4 + i[1] - neighbour_size:
    #                  1920 / 4 + i[1] + neighbour_size+1]
    #     mini_image = mini_image.reshape(
    #         1, neighbour_size*2+1, neighbour_size*2+1, 3)
    #
    #     if mini_images is None:
    #         mini_images = mini_image
    #     else:
    #         mini_images = np.r_[mini_images, mini_image]
    #
    #     if mini_images.shape[0]%10000==0:
    #         scipy.misc.imshow(mini_image)

# colored = image[1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi,
#                   1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi, :]
# colored[anomaly_single_label != 5] = np.array([0,0,0])
# scipy.misc.imshow(colored)



# scalers = []
#
# for i in range(anomaly_images.shape[1]):
#     scalers.append(StandardScaler())
#     anomaly_images[:,i] = scalers[i].fit_transform(anomaly_images[:,i])
#
# elm = ExtremeLearningMachine(NoMetric, iterations=1)
# elm.add_existing_layer(ELMAE(5))
# elm.add_existing_layer(ELMAE(5))
# elm.fit(anomaly_images)
# elm_anomaly = elm.predict(anomaly_images)
#
# elm_scalers = []
# for i in range(elm_anomaly.shape[1]):
#     elm_scalers.append(StandardScaler())
#     elm_anomaly[:,i] = elm_scalers[i].fit_transform(elm_anomaly[:,i])
#
# print(elm_anomaly.shape, np.min(elm_anomaly), np.max(elm_anomaly))
#
#
# kmeans = MiniBatchKMeans(n_clusters=k_cluster, random_state=0).fit(elm_anomaly)
# elm_labels = kmeans.labels_
#
# for j in range(0, 4):
#     elm_single_label = elm_labels[j*(w*h):(j+1)*(w*h)]
#     elm_single_label = np.reshape(elm_single_label, (w, h))
#     scipy.misc.imsave(u'elm_labels_{0:d}.jpg'.format(j+1),
#                       elm_single_label)
#
# elm_labels = np.reshape(elm_labels, (w, h))
# scipy.misc.imsave("elm_labels.jpg", elm_labels)
