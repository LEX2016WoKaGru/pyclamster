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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
image_dir = os.path.join(BASE_DIR, "examples", "images", "lex")
data_dir = os.path.join(BASE_DIR, "data")
trained_models = os.path.join(BASE_DIR, "data")

good_angle = 45
center = int(1920/2)
good_angle_dpi = int(np.round(1920 / 180 * good_angle))
denoising_ratio = 10

sessions = []
sessions.append(pickle.load(open(os.path.join(data_dir,'sessions/FE3_session_new.pk'),'rb')))
sessions.append(pickle.load(open(os.path.join(data_dir,'sessions/FE4_session_new.pk'),'rb')))

#session.set_images('/home/yann/Studium/LEX/LEX/kite/cam3/FE3*.jpg')
sessions[0].set_images(os.path.join(image_dir,'cam3/FE3_Image_20160901_100000_UTCp1.jpg'))
sessions[1].set_images(os.path.join(image_dir,'cam4/FE4_Image_20160901_100000_UTCp1.jpg'))

def simg(image):
    plt.figure()
    plt.imshow(image.data)
    plt.show()

kmeans = pickle.load(open(os.path.join(trained_models, "kmeans.pk"), "rb"))

clouds =[]
clobj = []
for s in sessions:
    for image in s.iterate_over_rectified_images():
        image_lcn = Image(image)
        image_lcn.data = LCN(size=(50,50,3), scale=False).fit_transform(image_lcn.data)
        
        w, h, _ = original_shape = image_lcn.data.shape
        raw_image_lcn = rbDetection(image_lcn.data).reshape((w*h, -1))
        #raw_image = image.data.reshape((w*h, -1))
        label = kmeans.predict(raw_image_lcn)
        label.reshape((w, h), replace=True)
        
        masks = label.getMaskStore()
        cloud_mask_num = [1]
        masks.denoise(cloud_mask_num, 1000) # cloud - sky choose right number (0 or 1)
        
        cloud_labels_object, numLabels = masks.wsMask(cloud_mask_num,np.ones((11,11))) 
        clobj.append(cloud_labels_object)
        # NOTE: there will be cloud-lables as well as 0 !!!
        print ("number of detected clouds = "+str(numLabels))
        
        cloud_store = cloud_labels_object.getMaskStore()
        cloud_lables = [l+1 for l in range(numLabels)]
        clouds.append([cloud_store.getCloud(image, [k,]) for k in cloud_lables])
            
plt.figure()
plt.subplot(121)
plt.imshow(clobj[0].labels)
plt.subplot(122)
plt.imshow(clobj[1].labels)
plt.show()

print("----starting cloud-cloud matching")
start = time.time()
m = pyclamster.matching.Matching()
matching_result,_ = m.matching(clouds[0],clouds[1])
gestime = time.time()-start
print('time = '+str(gestime))

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

