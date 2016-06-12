# -*- coding: utf-8 -*-
"""
Created on 06.06.16

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
import time

# External modules
import numpy as np
import tables

import scipy.misc

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from pyextremelm import ELMClassifier
from pyextremelm.builder import metrics
from pyextremelm.builder import ExtremeLearningMachine
from pyextremelm.builder import layers as ELMLayers

# Internal modules
from pyclamster.image import Image
from pyclamster.clustering.functions import localBrightness
from pyclamster.clustering.preprocess import ZCA

__version__ = ""

hdf5_save_path = "./training.hdf5"
n_neurons = 500
test_image = "./images/wettermast/Image_Wkm_Aktuell_2.jpg"
test_size = 0.1
patch_size = (21, 21)


compression_filter = tables.Filters(complevel=5, complib='blosc')
hdf5_file = tables.open_file(hdf5_save_path, filters=compression_filter,
                             mode="r", title="Trainings data")

# Get the data from HDF5
X = hdf5_file.root.wettermast.patches[:]
y = hdf5_file.root.wettermast.labels[:]

hdf5_file.close()


# Reshape the X vector as 2d Vector
X = np.reshape(X, (X.shape[0], -1))

# Split in trainings and test data
train_X = X[:int(len(y)*(1-test_size)), :]
train_y = y[:int(len(y)*(1-test_size))]


test_X = X[int(len(y)*(1-test_size)):, :]
test_y = y[int(len(y)*(1-test_size)):]

# Scale the X-Values
# whitening = ZCA()
# train_X = whitening.fit_transform(train_X)
# test_X = whitening.transform(test_X)
#
# elmae = ExtremeLearningMachine(metrics.MeanSquaredError, 1)
# elmae.add_existing_layer(ELMLayers.ELMAE(n_neurons, C=10E10))
# elmae.add_existing_layer(ELMLayers.ELMRandom(n_neurons))
# elmae.add_existing_layer(ELMLayers.ELMClRidge(output_neurons=1, C=10E10, weighted=True))

#elmae = ELMClassifier(n_neurons, C=10E20, iters=1, weighted=False)
elmae = AdaBoostClassifier()

elmae.fit(train_X, train_y)

prediction = elmae.predict(test_X)
#print(prediction)
print("Test accuracy: {0:f}".format(accuracy_score(test_y, prediction)))


# Load test image and create test samples
start_time = time.time()
img = Image(test_image)
cutted = img.cut((460, 460, 1460, 1460))

# Correct the local image brightness

patches = extract_patches_2d(cutted.data, patch_size)

patch_len = len(patches)
step_width = int(patch_len*0.5)

detection = None

input = None
for i in range(0, patch_len, step_width):
    mini_patch = patches[i:i+step_width]
    mini_patch = mini_patch.reshape((mini_patch.shape[0], -1))
    if detection is None:
        detection = elmae.predict(input)
    else:
        detection = np.concatenate(
            (detection, elmae.predict(input)), axis=0)

print(np.min(detection), np.max(detection), np.percentile(detection, 90))
detection = detection.reshape((1000-patch_size[0]+1, 1000-patch_size[1]+1))

scipy.misc.imsave("original.jpg", img.data[
    460+(patch_size[0]-1)/2:1460-(patch_size[0]-1)/2,
    460+(patch_size[1]-1)/2:1460-(patch_size[1]-1)/2])
scipy.misc.imsave("detection.png", detection)

print("finished after {0:f} seconds".format(time.time()-start_time))