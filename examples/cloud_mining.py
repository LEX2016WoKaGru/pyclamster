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
from copy import deepcopy

# External modules
import scipy
import scipy.misc
import numpy as np  # Standard import
import sklearn.cluster  # For k-means
from sklearn.feature_extraction.image import \
    extract_patches_2d  # To extract image patches
import tables  # For HDF5

# Internal modules
import pyclamster as pycl
from pyclamster.clustering.Labels import Labels
# from pyclamster.clustering.kmeans import KMeans
from pyclamster.clustering.functions import localBrightness, rbDetection, \
    listShuffleSplit, cloudPatchChecker

__version__ = "0.1"

"""
At the moment only a container for ideas!
Script to create the training dataset for the supervised cloud detection
algorithm with an unsupervised approach.

# TODO if it is working, check with big data and memory!
Possible new version of k-means necessary.
"""

directory = "./images"
k_cluster = 2

hdf5_save_path = "./training.hdf5"

concated_images = None
labels = None

all_images = []
temp_images = []
cluster = sklearn.cluster.MiniBatchKMeans(n_clusters=k_cluster, random_state=0)
compression_filter = tables.Filters(complevel=5, complib='blosc')
hdf5_file = tables.open_file(hdf5_save_path, filters=compression_filter,
                             mode="w", title="Trainings data")
used_labels = 0

all_images += glob.glob(os.path.join(directory, "wettermast", "*.jpg"))
# print(all_images)

for subdir, dirs, files in os.walk(directory):
    temp_images += glob.glob(os.path.join(subdir, "*.jpg"))

temp_images = listShuffleSplit(temp_images, 10)
all_images = [all_images] + temp_images

# all_images.append(temp_images)
for image_list in all_images:
    concated_images = None
    for image_path in image_list:
        # Read in the image
        img = pycl.Image(image_path)
        """
        Here comes some pre-process corrections, like projection correction etc.
        """
        # Normalize the image
        img.data = localBrightness(img.data, nh_size=10)
        img.crop((460, 460, 1460, 1460))
        # Blue and red cloud detection algorithm
        img_data = rbDetection(img.data)
        w, h = original_shape = tuple(img_data.shape)
        if concated_images is None:
            concated_images = np.reshape(img_data, (w * h, -1))
        else:
            concated_images = np.r_[concated_images,
                                    np.reshape(img_data, (w * h, -1))]

    start_time = time.time()
    # Check if k-means is already initialized
    # If yes, than update (partial) fit, if not full fit.
    try:
        cluster.labels_
        cluster.partial_fit(concated_images)
    except:
        cluster.fit(concated_images)
    print(concated_images.shape[0], time.time() - start_time)

    # Get the labels (Cloud = 1, No cloud = 0)
    labels = Labels(cluster.labels_ * (-1) + 1)

    # Split the labels into the original image parts
    splitted_labels = labels.splitUp(indices_or_sections=len(image_list))
    for key, splitted_label in enumerate(splitted_labels):
        targets = None
        mini_images = None

        # Reshape the labels into the width and height of the image
        label = splitted_label.reshape((w, h), replace=False)
        start_time = time.time()

        # Load the original image
        img = pycl.Image(image_list[key])
        cutted_img = img.cut((460, 460, 1460, 1460))

        # Add the label layer as fourth layer into the color layer.
        labeled_image = np.concatenate(
            (cutted_img.data, label.labels.reshape((w, h, 1))), axis=2)

        # Extract the image patches
        img_patches = extract_patches_2d(
            labeled_image, (31, 31), 100, random_state=42)
        for patch in img_patches:

            # Check if patches fulfill the requirements
            mini_image, value = cloudPatchChecker(patch, crit=0.95)

            # If fulfill, then expand the arrays
            if not mini_image is None:
                mini_image_size = tuple([1] + list(mini_image.shape))
                mini_image = np.reshape(mini_image, mini_image_size)
                if mini_images is None:
                    mini_images = mini_image
                    targets = np.array(value)
                else:
                    mini_images = np.concatenate((mini_images, mini_image),
                                                 axis=0)
                    targets = np.append(targets, value)

        ######################
        # Write data to hdf5 #
        ######################
        base_dir = os.path.split(os.path.dirname(image_list[key]))[-1]
        # Check if files are already set
        try:
            a = patches_storage[0]
        except:
            image_storage = hdf5_file.createEArray(
                "/{0:s}".format(base_dir), 'image',
                tables.Atom.from_dtype(cutted_img.data.dtype),
                shape=(0, cutted_img.data.shape[0], cutted_img.data.shape[1],
                       cutted_img.data.shape[2]),
                filters=compression_filter, createparents=True)
            labeled_storage = hdf5_file.createEArray(
                "/{0:s}".format(base_dir), 'labeled_image',
                tables.Atom.from_dtype(label.labels.dtype),
                shape=(0, label.labels.shape[0], label.labels.shape[1]),
                filters=compression_filter, createparents=True)
            time_storage = hdf5_file.createEArray(
                "/{0:s}".format(base_dir), 'time_container',
                tables.Int64Atom(), shape=(0, 2), filters=compression_filter)
            labels_storage = hdf5_file.createEArray(
                "/{0:s}".format(base_dir), 'labels',
                tables.Atom.from_dtype(targets.dtype),
                shape=(0, 1), createparents=True)
            patches_storage = hdf5_file.createEArray(
                "/{0:s}".format(base_dir), 'patches',
                tables.Atom.from_dtype(mini_images.dtype),
                shape=(0, mini_image.shape[1], mini_image.shape[2], mini_image.shape[3]),
                createparents=True)

        # Write arrays to hdf5
        image_storage.append(cutted_img.data.reshape(
            (1, cutted_img.data.shape[0], cutted_img.data.shape[1], cutted_img.data.shape[2])))
        labeled_storage.append(label.labels.reshape(
            (1, label.labels.shape[0], label.labels.shape[1])))
        labels_storage.append(targets.reshape((targets.shape[0], 1)))
        patches_storage.append(mini_images)
        time_length = np.array([time_storage[-1, 1], time_storage[-1, 1] + len(targets)])
        print(time_length.reshape((-1,2)).shape)
        time_storage.append(time_length.reshape((-1,2)))

        print(image_list[key], len(targets), time.time() - start_time)

hdf5_file.close()
