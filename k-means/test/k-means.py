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
wm = scipy.ndimage.imread("Image_Wkm_Aktuell.jpg", mode="RGB")
# Add other dimension beside RGB
# rgb_mean = wm.copy()
# num_neighbor=1
# for index, val in np.ndenumerate(rgb_mean):
#         x = [index[0]-1, index[0]+1]
#         y = [index[1]-1, index[1]+1]
#         rgb_mean[index] = np.mean()
wm = wm[1920/2-good_angle_dpi:1920/2+good_angle_dpi,
        1920/2-good_angle_dpi:1920/2+good_angle_dpi, :]
scipy.misc.imsave("test_original_3.jpg", wm)

wm = wm/255

w, h, d = original_shape = tuple(wm.shape)
assert d == 3
image_array = np.reshape(wm, (w * h, d))
t0 = time()
kmeans = KMeans(n_clusters=k_cluster, random_state=0).fit(image_array)
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))
coloured_labels = labels/2
wm_labeled = np.reshape(coloured_labels, (w, h))

scipy.misc.imsave("test_labeled_3.jpg", wm_labeled)
