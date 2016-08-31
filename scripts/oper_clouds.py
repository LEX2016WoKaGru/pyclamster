# -*- coding: utf-8 -*-
"""
Created on 14.08.16

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
import warnings
import pickle

# External modules
import scipy.misc

# Internal modules
from pyclamster import Image, Camera
from pyclamster.clustering.preprocess import LCN
from pyclamster.clustering.functions import rbDetection

warnings.catch_warnings()
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

image_directory = os.path.join(BASE_DIR, "examples", "images", "wolf")
trained_models = os.path.join(BASE_DIR, "data")

good_angle = 45
center = int(1920/2)
good_angle_dpi = int(np.round(1920 / 180 * good_angle))


predictor = pickle.load(open(os.path.join(trained_models, "kmeans.pk"), "rb"))

for image_path in all_images:
    # TODO: Here is the 'entzerrung' missing
    image.data = LCN(size=(50,50,3), scale=False).fit_transform(image.data/256)
    raw_image = rbDetection(image.data)
    w, h = original_shape = tuple(raw_image[:, :].shape)
    raw_image = np.reshape(raw_image, (w * h, 1))
    label = predictor.predict(raw_image)
    label.reshape((960, 960), replace=True)
    scipy.misc.imsave("cloud.png", label.labels)
    masks = label.getMaskStore()
    masks.denoise([0], 960)
    cloud_labels, _ = masks.labelMask([0,])
    scipy.misc.imsave("labels.png", cloud_labels.labels)
    scipy.misc.imshow(cloud_labels.labels)
    cloud_store = cloud_labels.getMaskStore()
    # TODO: Here is the matching algorithm missing



