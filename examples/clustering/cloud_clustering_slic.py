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

# External modules
import numpy as np
import scipy.interpolate
import scipy.misc
import scipy.ndimage

from skimage.segmentation import slic
from skimage.filters import threshold_otsu
from skimage.filters import rank
from skimage.morphology import disk
from skimage.color import rgb2hsv
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters
from skimage import exposure



import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Internal modules
from pyclamster import Image, Labels
from pyclamster.clustering.preprocess import LCN
from pyclamster.functions import rbDetection

plt.style.use('typhon')

warnings.catch_warnings()
warnings.filterwarnings('ignore')


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
image_directory = os.path.join(BASE_DIR, "examples", "images", 'wettermast')
trained_models = os.path.join(BASE_DIR, "data")

good_angle = 45
center = int(1920/2)
good_angle_dpi = int(np.round(1920 / 180 * good_angle))
denoising_ratio = 10
all_images = glob.glob(os.path.join(image_directory, "Image_*.jpg"))


predictor = pickle.load(open(os.path.join(trained_models, "kmeans.pk"), "rb"))

for image_path in all_images:
    image = Image(image_path)
    image.cut([center - good_angle_dpi, center-good_angle_dpi, center+good_angle_dpi, center + good_angle_dpi]).save('test.jpg')
    image.data = image.data[center - good_angle_dpi:center + good_angle_dpi,
                            center - good_angle_dpi:center + good_angle_dpi]
    #selem = np.ones((50,50))
    #image.data = equal_each(image.data, selem)
    #segmented_image = slic(image.data, n_segments=50, compactness=10, sigma=1)+1
    segmented_image = slic(image.data, slic_zero=True)+1
    rb_image = image.data[:,:,0]-image.data[:,:,2]
    #selem = np.ones((250, 250))
    #rb_image = exposure.adjust_sigmoid(rb_image, cutoff=0.1, gain=0.5)
    #p2, p98 = np.percentile(rb_image, (2, 98))
    #rb_image = exposure.rescale_intensity(rb_image, in_range=(p2, p98))
    #rb_image = exposure.equalize_adapthist(rb_image, clip_limit=0.03)*255
    # rb_image = rb_image-np.min(rb_image)
    # rb_image = rb_image/(np.max(rb_image)-np.min(rb_image)+0.1)
    # rb_image = rb_image*255
    #p10, p90 = np.percentile(rb_image, (10, 90))
    global_thres = threshold_otsu(
        rb_image[rb_image>10])
    threshold_array = np.zeros_like(segmented_image)
    threshold_array[:] = global_thres
    # threshold_array[:] = np.nan
    # for label in np.unique(segmented_image):
    #     masked_rb = np.ma.masked_where(segmented_image!=label, rb_image)
    #     lcenter = segmented_image==label
    #     if (masked_rb.max()<global_thres) or (masked_rb.min()>global_thres):
    #         threshold_array[lcenter] = global_thres
    #     else:
    #         local_otsu = threshold_otsu(rb_image[segmented_image==label])
    #         threshold_array[lcenter] = 0.5*local_otsu+0.5*global_thres
    # threshold_array = scipy.ndimage.filters.maximum_filter(threshold_array, footprint=np.ones((40,40)), mode='nearest')
    # threshold_array = scipy.ndimage.filters.gaussian_filter(threshold_array, sigma=20, mode='nearest')
    label = Labels((np.logical_or(rb_image>threshold_array, rb_image<10)).astype(int))
    scipy.misc.imsave("cloud.png", label.labels)
    masks = label.getMaskStore()
    masks.denoise([1], 2000)
    cloud_labels, _ = masks.givenLabelMask(segmented_image, [1,])
    scipy.misc.imsave("labels.png", cloud_labels.labels)
    gs = gridspec.GridSpec(2,3)
    ax = plt.subplot(gs[0, 0])
    ax.imshow(image.data)
    ax = plt.subplot(gs[0, 1])
    ax.imshow(rb_image)
    ax = plt.subplot(gs[1, :2])
    ax.hist(rb_image.reshape((-1)), bins=256)
    ax.axvline(x=global_thres)
    ax = plt.subplot(gs[1, 2])
    ax.imshow(threshold_array)
    ax = plt.subplot(gs[0, 2])
    ax.imshow(cloud_labels.labels)
    plt.show()
    #scipy.misc.imshow(cloud_labels.labels)
    cloud_store = cloud_labels.getMaskStore()
