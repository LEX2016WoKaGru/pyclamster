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

from pyextremelm.builder import ExtremeLearningMachine
from pyextremelm.metrics import NoMetric
from pyextremelm.builder.layers.unsupervised import ELMAE
from pyextremelm.builder import training

# Internal modules


__version__ = "0.1"

k_cluster = 3
good_angle = 45
good_angle_dpi = int(np.round(1920 / 180 * good_angle))
receptive_size = 50
pooling_size = 30
n_hidden_layers = 10

pooled_layers = None
anomaly_images = None


def calc_feature_mapping(X):
    X = np.reshape(X, (X.shape[0], 1))
    feature_layer = training.random.ELMOrthoRandom(1, bias=False)
    return np.sum(feature_layer.fit(X))

image = scipy.ndimage.imread(u'Image_Wkm_Aktuell_3.jpg', mode="RGB")

scipy.misc.imsave(u'pooled_layer/image.jpg', image)

hidden_layers = []

for j in range(n_hidden_layers):
    hidden_layers.append(
        scipy.ndimage.filters.generic_filter(
            image, calc_feature_mapping,
            size=(receptive_size, receptive_size, 1), mode="constant"))
    print(u'finished {0:d} feature mapping'.format(j+1))

for key, layer in enumerate(hidden_layers):
    pooled_layer = np.sqrt(scipy.ndimage.filters.uniform_filter(
        layer**2, size=(pooling_size, pooling_size, 1), mode="constant"))

    anomaly_image = image - pooled_layer

    pooled_layer = pooled_layer [
                   1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi,
                   1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi, :]
    scipy.misc.imsave(u'pooled_layer/pooled_{0:d}.jpg'.format(key),
                      pooled_layer)

    w, h, d = tuple(pooled_layer.shape)
    pooled_layer = np.reshape(pooled_layer, (w * h, d))
    if pooled_layers is None:
        pooled_layers = pooled_layer
    else:
        pooled_layers = np.c_[pooled_layers, pooled_layer]

    anomaly_image = anomaly_image[
                    1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi,
                    1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi, :]
    scipy.misc.imsave(u'pooled_layer/anomaly_{0:d}.jpg'.format(key),
                      anomaly_image)

    w, h, d = tuple(anomaly_image.shape)
    anomaly_image = np.reshape(anomaly_image, (w * h, d))
    if anomaly_images is None:
        anomaly_images = anomaly_image
    else:
        anomaly_images = np.c_[anomaly_images, anomaly_image]

    print(u'finished {0:i} pooling'.format(key+1))


pickle.dump(pooled_layers, open("pooled.p", "wb"))
pickle.dump(anomaly_images, open("anomaly.p", "wb"))


kmeans = MiniBatchKMeans(n_clusters=k_cluster, random_state=0).fit(
    pooled_layers)
pooled_labels = kmeans.labels_
pooled_labels = np.reshape(pooled_labels, (w, h))
scipy.misc.imsave(u'pooled_labels.jpg', pooled_labels)


kmeans = MiniBatchKMeans(n_clusters=k_cluster, random_state=0).fit(
    anomaly_images)
anomaly_labels = kmeans.labels_
anomaly_labels = np.reshape(anomaly_labels, (w, h))
scipy.misc.imsave(u'anomaly_labels.jpg', anomaly_labels)


# for i in range(0, 4):
#     anomaly_single_label = anomaly_labels[i*(w*h):(i+1)*(w*h)]
#     anomaly_single_label = np.reshape(anomaly_single_label, (w, h))
#     scipy.misc.imsave(u'anomaly_labels_{0:d}.jpg'.format(i+1),
#                       anomaly_single_label)
# #
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
# #
# elm_labels = np.reshape(elm_labels, (w, h))
# scipy.misc.imsave("elm_labels.jpg", elm_labels)
