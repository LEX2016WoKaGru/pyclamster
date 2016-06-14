# -*- coding: utf-8 -*-
"""
Created on 27.05.16

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
import pickle
import warnings
import glob
import os

# External modules
import numpy as np
import scipy.ndimage

# Internal modules
from pyclamster import Image
from pyclamster.clustering.preprocess import LCN
from pyclamster.clustering.kmeans import KMeans
from pyclamster.clustering.functions import rbDetection

warnings.catch_warnings()
warnings.filterwarnings('ignore')

__version__ = "0.1"

base_folder = "/home/tfinn/Projects/pyclamster/"
image_directory = os.path.join(base_folder, "examples", "images", "wettermast")
trained_models = os.path.join(base_folder, "trained_models")

k_cluster = 2
good_angle = 45
center = int(1920/2)
good_angle_dpi = int(np.round(1920 / 180 * good_angle))
denoising_ratio = 10
all_images = glob.glob(os.path.join(image_directory, "*.jpg"))

anomaly_images = None

for image_path in all_images:
    image = Image(image_path)
    image.data = LCN(size=(50,50,3), scale=False).fit_transform(image.data)
    image.data = image.data[center - good_angle_dpi:center + good_angle_dpi,
                            center - good_angle_dpi:center + good_angle_dpi]
    anomaly_image = rbDetection(image.data)
    w, h = original_shape = tuple(anomaly_image.shape)
    anomaly_image = np.reshape(anomaly_image, (w * h, -1))
    if anomaly_images is None:
        anomaly_images = anomaly_image
    else:
        anomaly_images = np.r_[anomaly_images, anomaly_image]

kmeans = KMeans(2).fit(anomaly_images)

pickle.dump(kmeans, open(os.path.join(trained_models, "kmeans.pk"), "wb"))