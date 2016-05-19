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
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from time import time

import pyextremelm

k_cluster = 5
good_angle = 45
good_angle_dpi = int(np.round(1920/180*good_angle))

# Load the Wettermast photo
wm = scipy.ndimage.imread("Image_Wkm_Aktuell_3.jpg", mode="RGB")
wm = wm/255 # to get values between 0 and 1
# Normalize RGB to surrounding fields
neighbour_size = 100
rgb_mean = scipy.ndimage.filters.uniform_filter(
    wm, size=(neighbour_size, neighbour_size, 1), mode="constant")

#test = wm-rgb_mean

# rgb_median = scipy.ndimage.filters.median_filter(
#     wm, size=(neighbour_size, neighbour_size, 1), mode="constant")

# rgb_std = scipy.ndimage.filters.generic_filter(
#     wm, function=np.std, size=(neighbour_size, neighbour_size, 1),
#     mode="constant")

print("success")
# #print(rgb_mean)
rgb_sqr_mean = scipy.ndimage.uniform_filter(
    wm**2,size=(neighbour_size, neighbour_size, 1), mode="constant")
# #print(rgb_sqr_mean)
rgb_stddev = rgb_sqr_mean - rgb_mean**2
rgb_stddev[rgb_stddev<0] = 0
rgb_stddev = np.sqrt(rgb_stddev)
print(np.max(rgb_stddev), np.min(rgb_stddev))
#rgb_standard = (wm-rgb_mean)
# rgb_std_max = scipy.ndimage.maximum_filter(
#     wm,size=(neighbour_size, neighbour_size, 1), mode="constant")
# rgb_std_min = scipy.ndimage.minimum_filter(
#     wm,size=(neighbour_size, neighbour_size, 1), mode="constant")
# rgb_minmax = rgb_standard/(rgb_std_max-rgb_std_min)
rgb_standard = (wm-rgb_mean)/rgb_stddev
rgb_standard[rgb_stddev<0.001] = 0
rgb_standard = rgb_standard[1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi,
     1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi, :]

scipy.misc.imsave(
 "test_mean.jpg", (rgb_standard))

neighbour_size = 10
rgb_mean = scipy.ndimage.filters.uniform_filter(
    rgb_standard, size=(neighbour_size, neighbour_size, 1), mode="constant")
rgb_sqr_mean = scipy.ndimage.uniform_filter(
    rgb_standard**2,size=(neighbour_size, neighbour_size, 1), mode="constant")
# #print(rgb_sqr_mean)
rgb_stddev = rgb_sqr_mean - rgb_mean**2
rgb_stddev[rgb_stddev<0] = 0
rgb_stddev = np.sqrt(rgb_stddev)

wm = wm[
               1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi,
               1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi, :]

wm[rgb_stddev>0.5] = 1

scipy.misc.imsave(
 "test_std.jpg", (wm))

#print(rgb_mean.shape)

# # wm_added_dim = np.concatenate((wm, rgb_mean, rgb_stddev), axis=2)
# # print(wm_added_dim.shape)
# wm = wm[1920/2-good_angle_dpi:1920/2+good_angle_dpi,
#         1920/2-good_angle_dpi:1920/2+good_angle_dpi, :]
# # wm_added_dim = wm_added_dim[1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi,
# #     1920 / 2 - good_angle_dpi:1920 / 2 + good_angle_dpi, :]
# scipy.misc.imsave(
#     "test_original.jpg", (wm*255))
# appendix = ["r", "g", "b"]
# for i in range(0, wm.shape[2]):
#     scipy.misc.imsave("test_original_%s.jpg" % (appendix[i]),
#                       (wm * 255)[:, :, i])
#
# #wm = wm/255
#
# w, h, d = original_shape = tuple(wm.shape)
# image_array = np.reshape(wm, (w * h, d))
# t0 = time()
# kmeans = MiniBatchKMeans(n_clusters=k_cluster, random_state=0).fit(image_array)
# labels = kmeans.labels_
# print("done in %0.3fs." % (time() - t0))
# #coloured_labels = labels/2
# wm_labeled = np.reshape(labels, (w, h))
#
# scipy.misc.imsave("test_labeled_kmeans.jpg", wm_labeled)
# # #
# image_scale = StandardScaler()
# test = image_scale.fit_transform(image_array)
# dbscan_inst = DBSCAN(eps=1, min_samples=10000)
# #labels = dbscan_inst.fit_predict(test)
# print("done in %0.3fs." % (time() - t0))
# #coloured_labels = labels/2
# wm_labeled = np.reshape(labels, (w, h))
# scipy.misc.imsave("test_labeled_dbscan.jpg", wm_labeled)
# t0 = time()
# labels = DBSCAN().fit_predict(image_array)
# #labels = db.predict(image_array)
# print("done in %0.3fs." % (time() - t0))
# #coloured_labels = labels/2
# wm_labeled = np.reshape(labels, (w, h))
#
# scipy.misc.imsave("test_labeled_dbscan.jpg", wm_labeled)