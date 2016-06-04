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

# External modules
import numpy as np

# Internal modules
import pyclamster as pycl
#from pyclamster.clustering.base import Labels
#from pyclamster.clustering.kmeans import KMeans
from pyclamster.clustering.preprocess import LocalBrightness, RBDetection

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

for image_path in all_images:
    img = pycl.Image()
    img.loadImage(image_path)
    """
    Here comes some pre-process corrections, like projection correction etc.
    """
    img_mean = LocalBrightness().fit_transform(img)
    img_rb = RBDetection().fit_transform(img_mean)
    flatted_img = img_rb.image[:,:,0::2]
    w, h, d = original_shape = tuple(flatted_img.shape)
    if concated_images is None:
        concated_images = np.reshape(flatted_img, (w * h, d))
    else:
        concated_images = np.r_[concated_images,
                                np.reshape(flatted_img, (w * h, d))]
print(concated_images)
# kmeans = KMeans(n_clusters=k_cluster, random_state=0).fit(concated_images)
# labels = kmeans.labels_
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

