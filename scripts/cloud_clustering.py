# -*- coding: utf-8 -*-
"""
Created on 13.06.16

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
import glob
import os

# External modules
import numpy as np
import scipy.misc
import scipy.ndimage

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from skimage.segmentation import random_walker
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import morphology



# Internal modules
from pyclamster import Image
from pyclamster.clustering.preprocess import LCN, ZCA
from pyclamster.clustering.kmeans import KMeans
from pyclamster.clustering.functions import localBrightness, rbDetection

warnings.catch_warnings()
warnings.filterwarnings('ignore')


__version__ = ""

base_folder = "/home/tfinn/Projects/pyclamster/"
image_directory = os.path.join(base_folder, "examples", "images", "wettermast")
trained_models = os.path.join(base_folder, "trained_models")

good_angle = 45
center = int(1920/2)
good_angle_dpi = int(np.round(1920 / 180 * good_angle))
denoising_ratio = 10
all_images = glob.glob(os.path.join(image_directory, "*.jpg"))
all_images = [os.path.join(image_directory, "Image_Wkm_Aktuell_2.jpg"),]



kmeans = pickle.load(open(os.path.join(trained_models, "kmeans.pk"), "rb"))

for image_path in all_images:
    image = Image(image_path)
    image.data = LCN(size=(50,50,3), scale=False).fit_transform(image.data)
    image.data = image.data[center - good_angle_dpi:center + good_angle_dpi,
                 center - good_angle_dpi:center + good_angle_dpi]
    w, h, _ = original_shape = image.data.shape
    raw_image = rbDetection(image.data).reshape((w*h, -1))
    label = kmeans.predict(raw_image)
    label.reshape((w, h), replace=True)
    masks = label.getMaskStore()
    masks.denoise([0], 1000)
    #scipy.misc.imshow(masks.getMask([0,]))
    image = masks.applyMask(image, [0,])
    labels, _ = masks.labelMask([0,])
    scipy.misc.imshow(labels.labels)
    # w, h, n = original_shape = tuple(image.data.shape)
    # anomaly_image = np.reshape(image.data, (w * h, n))
    # fill_value = anomaly_image.get_fill_value()
    # print("before ", anomaly_image.shape)
    # anomaly_image = anomaly_image[anomaly_image.mask.any(axis=1)]
    # print("after ", anomaly_image.shape)
    # #print(KMeans().bestK(anomaly_image))