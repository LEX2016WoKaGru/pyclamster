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
import time

# External modules
import numpy as np
import scipy.ndimage

# Internal modules
from pyclamster import Image
from pyclamster.clustering.preprocess import LCN
from pyclamster.clustering.kmeans import KMeans
from pyclamster.functions import rbDetection

warnings.catch_warnings()
warnings.filterwarnings('ignore')

__version__ = "0.1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
image_directory = os.path.join(BASE_DIR, "examples", "images", "wettermast")
trained_models = os.path.join(BASE_DIR, "data")
batch_dir = None # Folder for the batch mode

k_cluster = 2
good_angle = 45
center = int(1920/2)
good_angle_dpi = int(np.round(1920 / 180 * good_angle))

batch_size=50
nb_batches = 100

all_images = glob.glob(os.path.join(image_directory, "*.jpg"))

anomaly_images = None

for image_path in all_images:
    image = Image(image_path)
    image.data = LCN(size=(50,50,3), scale=False).fit_transform(image.data)
    image.data = image.data[center - good_angle_dpi:center + good_angle_dpi,
                            center - good_angle_dpi:center + good_angle_dpi]
    anomaly_image = rbDetection(image.data)
    w, h = original_shape = tuple(anomaly_image[:, :].shape)
    anomaly_image = np.reshape(anomaly_image, (w * h, 1))
    anomaly_image[anomaly_image < -10000] = -10000
    anomaly_image[anomaly_image > 10000] = 10000
    if anomaly_images is None:
        anomaly_images = anomaly_image
    else:
        anomaly_images = np.r_[anomaly_images, anomaly_image]
kmeans = KMeans(2).fit(anomaly_images)
print('Finished intital clustering')

# Batch wise cluster update with a given archive of cloud images.
if not batch_dir is None:
    # Needed if batch mode should be used
    from keras.preprocessing.image import ImageDataGenerator
    data_gen = ImageDataGenerator(
        rescale=1./256., zoom_range=(0.5, 0.5), dim_ordering='tf')
    image_generator = data_gen.flow_from_directory(batch_dir,
                                 class_mode=None, shuffle=True, seed=42,
                                 color_mode='rgb', batch_size=batch_size)
    n_batch = 0
    eta_time = []
    for x_batch in image_generator:
        start_time = time.time()
        processed_img = np.empty((batch_size,256,256))
        for i in range(x_batch.shape[0]):
            x_batch[i] = LCN(size=(50,50,3), scale=False).fit_transform(x_batch[i])
            processed_img[i] = rbDetection(x_batch[i])
        processed_img = processed_img.reshape((-1, 1))
        processed_img[processed_img<-10000] = -10000
        processed_img[processed_img>10000] = 10000
        kmeans = kmeans.partial_fit(processed_img)
        eta_time.append(time.time()-start_time)
        if n_batch>=nb_batches:
            break
        n_batch += 1
        print('{0:d}/{1:d}, eta {2:.1f} s'.format(
            n_batch, nb_batches, (nb_batches-n_batch)*np.mean(eta_time)))

pickle.dump(kmeans, open(os.path.join(trained_models, "kmeans.pk"), "wb"))
