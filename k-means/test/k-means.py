# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:14:13 2016

@author: tfinn
"""
# External modules

# Internal modules


__version__ = ""

print(__doc__)
import numpy as np
import scipy.misc
import scipy.ndimage
from sklearn.cluster import KMeans
from time import time

k_cluster = 4
good_angle = 45
good_angle_dpi = int(np.round(1920/180*good_angle))

# Load the Wettermast photo
wm = scipy.ndimage.imread("Image_Wkm_Aktuell_2.jpg", mode="RGB")
# wm = wm/255 # to get values between 0 and 1
# Normalize RGB to surrounding fields
neighbour_size = 5
rgb_mean = scipy.ndimage.filters.uniform_filter(
    wm, size=(neighbour_size, neighbour_size, 1), mode="constant")
#print(rgb_mean)
rgb_sqr_mean = scipy.ndimage.uniform_filter(
    wm**2,size=(neighbour_size, neighbour_size, 1), mode="constant")
#print(rgb_sqr_mean)
rgb_stddev = np.sqrt(rgb_sqr_mean - rgb_mean**2)
wm_normal = (wm-rgb_mean)/rgb_stddev
wm_normal[rgb_stddev==0] = 0
#wm = np.concatenate((wm, rgb_mean), axis=2)
print(wm_normal.shape)
wm = wm[1920/2-good_angle_dpi:1920/2+good_angle_dpi,
        1920/2-good_angle_dpi:1920/2+good_angle_dpi, :]
wm_normal = wm_normal[1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi,
    1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi, :]
scipy.misc.imsave(
    "test_original.jpg", np.array([(wm_normal[:, :, i]*wm[:, :, i].std())
                                   +wm[:, :, i].mean()
                                   for i in range(wm.shape[2])]
                                  ))

#wm = wm/255

w, h, d = original_shape = tuple(wm_normal.shape)
image_array = np.reshape(wm_normal, (w * h, d))
t0 = time()
kmeans = KMeans(n_clusters=k_cluster, random_state=0).fit(image_array)
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))
coloured_labels = labels/2
wm_labeled = np.reshape(coloured_labels, (w, h))

scipy.misc.imsave("test_labeled.jpg", wm_labeled)
