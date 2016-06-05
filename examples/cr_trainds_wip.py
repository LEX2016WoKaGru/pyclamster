# -*- coding: utf-8 -*-
"""
Created on 02.06.16

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
import os
import glob
import pickle
import time

# External modules
import scipy
import numpy as np
import sklearn.cluster

# Internal modules
import pyclamster as pycl
from pyclamster.clustering.Labels import Labels
#from pyclamster.clustering.kmeans import KMeans
from pyclamster.clustering.functions import localBrightness, rbDetection,\
    listShuffleSplit

__version__ = "doesnt working!"

"""
At the moment only a container for ideas!
Script to create the training dataset for the supervised cloud detection
algorithm with an unsupervised approach.

# TODO if it is working, check with big data and memory!
Possible new version of k-means necessary.
"""

directory = "./images/stereo"
all_images = glob.glob(os.path.join(directory, "*.jpg"))
k_cluster = 2

concated_images = None
labels = None

targets = []
mini_images= []
cluster = sklearn.cluster.MiniBatchKMeans(n_clusters=k_cluster, random_state=0)

all_images = listShuffleSplit(all_images, 10)
for image_list in all_images:
    for image_path in image_list:
        img = pycl.Image()
        img.loadImage(image_path)
        """
        Here comes some pre-process corrections, like projection correction etc.
        """
        img.data = localBrightness(img.data)
        img.data = rbDetection(img.data)
        img.crop((480, 480, 1440, 1440))
        w, h = original_shape = tuple(img.data.shape)
        if concated_images is None:
            concated_images = np.reshape(img.data, (w * h, 1))
        else:
            concated_images = np.r_[concated_images,
                                    np.reshape(img.data, (w * h, 1))]
    #print(concated_images.shape)
    start_time = time.time()
    cluster.partial_fit(concated_images)
    #print(concated_images.shape[0], time.time()-start_time)
    labels = Labels(cluster.predict(concated_images))

    splitted_labels = labels.splitUp(indices_or_sections=len(image_list))
    for key, label in enumerate(splitted_labels):
        label.reshape((w, h), replace=True)
        label.filterRelevants(10, True)
        scipy.misc.imsave("%d.jpg" % key, label.labels)
#
# for key, path in enumerate(getImages(directory)):
#     single_label = Labels(labels=np.reshape(
#         labels[key * (w * h):(key + 1) * (w * h)], (w, h)))
#     """
#     #TODO Post process im_label to get only relevant labels
#     e.g. single_label.filterRelevants(neighbourhood_size=5)
#     """
#     positions, y = single_label.getSamples()
#     targets.append(y)
#     img = pycl.Image()
#     img.loadImage(path)
#     for coord in positions:
#         mini_images.append(img.nbCrop(coord, neighbourhood_size=10))
#
# training_data = {"X": mini_images, "y": targets}
# pickle.dump(training_data, open("training.p", "wb"))

