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
from pyclamster.clustering.old_labels import Labels
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
patches_per_image = 1000
hdf5_save_path = "./training.hdf5"
patch_size = (21, 21)

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
        img.data = localBrightness(img.data, (20, 20, 3))
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
        infos = None

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
            labeled_image, patch_size, patches_per_image, random_state=42)

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
            labels_storage = hdf5_file.createEArray(
                "/{0:s}".format(base_dir), 'labels',
                tables.Atom.from_dtype(targets.dtype),
                shape=(0, 1), createparents=True)
            patches_storage = hdf5_file.createEArray(
                "/{0:s}".format(base_dir), 'patches',
                tables.Atom.from_dtype(mini_images.dtype),
                shape=(0, mini_images.shape[1], mini_images.shape[2], mini_images.shape[3]),
                createparents=True)

        # Write arrays to hdf5
        labels_storage.append(targets.reshape((targets.shape[0], 1)))
        patches_storage.append(mini_images)

        print(image_list[key], len(targets), time.time() - start_time)

hdf5_file.close()
