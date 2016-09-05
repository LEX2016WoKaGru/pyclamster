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
import time
import copy

# External modules
import numpy as np
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt

#from sklearn.cluster import MiniBatchKMeans
#from sklearn.preprocessing import StandardScaler

from skimage.feature import match_template
from skimage.segmentation import random_walker
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import morphology



# Internal modules
import pyclamster
from pyclamster import Image
from pyclamster.matching.cloud import Cloud
from pyclamster.clustering.preprocess import LCN, ZCA
from pyclamster.clustering.kmeans import KMeans
from pyclamster.functions import localBrightness, rbDetection

warnings.catch_warnings()
warnings.filterwarnings('ignore')


__version__ = ""

base_folder = "./"
image_directory = os.path.join(base_folder, "examples", "images", "wolf")
trained_models = os.path.join(base_folder, "data")

good_angle = 45
center = int(1920/2)
good_angle_dpi = int(np.round(1920 / 180 * good_angle))
denoising_ratio = 10
#all_images = [
#    os.path.join(image_directory, "Image_20160531_114000_UTCp1_3.jpg"),
#    os.path.join(image_directory, "Image_20160531_114000_UTCp1_4.jpg")]
all_images = [
    os.path.join(base_folder, "examples", "images", "wettermast", "Image_Wkm_Aktuell_2.jpg"),
    os.path.join(base_folder, "examples", "images", "wettermast", "Image_Wkm_Aktuell_2.jpg")]


kmeans = pickle.load(open(os.path.join(trained_models, "kmeans.pk"), "rb"))

image = Image(all_images[0])
image_lcn = Image(image)
image_lcn.data = LCN(size=(50,50,3), scale=False).fit_transform(image_lcn.data)
image_lcn = image_lcn.cut([480, 480, 1480, 1480])
image_lcn.data = scipy.misc.imresize(image_lcn.data, (256,256), interp='bicubic')
image = image.cut([480,480,1480,1480])
image.data = scipy.misc.imresize(image.data, (256,256), interp='bicubic')

w, h, _ = original_shape = image_lcn.data.shape
raw_image_lcn = rbDetection(image_lcn.data).reshape((w*h, -1))
#raw_image = image.data.reshape((w*h, -1))
label = kmeans.predict(raw_image_lcn)
label.reshape((w, h), replace=True)

masks = label.getMaskStore()
cloud_mask_num = [1]
masks.denoise(cloud_mask_num, 1000) # cloud - sky choose right number (0 or 1)

cloud_labels_object, numLabels = masks.labelMask(cloud_mask_num) 
# NOTE: there will be cloud-lables as well as 0 !!!
print ("number of detected clouds = "+str(numLabels))

cloud_store = cloud_labels_object.getMaskStore()
cloud_lables = [l+1 for l in range(numLabels)]
clouds = [cloud_store.getCloud(image, [k,]) for k in cloud_lables] 
template = clouds[0].image


image2 = Image(all_images[0])
image2 = image2.cut([480, 480, 1480, 1480])
image2.data = scipy.misc.imresize(image2.data, (256,256), interp='bicubic')

print("----starting cloud-cloud matching")
start = time.time()
m = pyclamster.matching.Matching()
matching_result,_ = m.matching(clouds,clouds)
gestime = time.time()-start
print('time = '+str(gestime))

if 0: #save png's
    scipy.misc.imsave('image2.png', image2.data)
    scipy.misc.imsave('template_cloud.png', template.data)
    scipy.misc.imsave("lables_kmean.png", label.labels)
    scipy.misc.imsave("lables_used.png", cloud_labels_object.labels)
print('--- matched clouds = '+str(len(matching_result)))
for mapcl in matching_result: 
    spcl = mapcl[1]
    plt.figure()
    plt.subplot(131)
    plt.imshow(spcl.clouds[0].image.data)
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(spcl.prob_map.prob_map,cmap='BrBG')
    plt.colorbar()
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(spcl.clouds[1].image.data)
    plt.axis('off')
    plt.show()

print('------ results ----')
print(str(matching_result))
