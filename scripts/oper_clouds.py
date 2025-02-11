# -*- coding: utf-8 -*-
"""
Created on 14.08.16

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
import os
import warnings
import pickle
import glob

# External modules
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

# Internal modules
import pyclamster
from pyclamster.clustering.preprocess import LCN
from pyclamster.functions import rbDetection
import pyclamster.matching

warnings.catch_warnings()
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

image_directory = os.path.join(BASE_DIR, "examples", "images", "lex")
trained_models = os.path.join(BASE_DIR, "data")
plot_dir = os.path.join(BASE_DIR, 'plots')

predictor = pickle.load(open(os.path.join(trained_models, "kmeans.pk"), "rb"))

cams = []
cams.append(pickle.load(open(os.path.join(trained_models, 'sessions', 'FE3_session.pk'), mode='rb')))
cams.append(pickle.load(open(os.path.join(trained_models, 'sessions', 'FE4_session.pk'), mode='rb')))
cams[0].image_series = []
cams[1].image_series = []
cams[0].add_images(os.path.join(image_directory, 'cam3'))
cams[1].add_images(os.path.join(image_directory, 'cam4'))
matching = pyclamster.matching.Matching()

times = {'3': [], '4': []}
# load times
for img3 in cams[0]:
    img3.loadTimefromfilename('FE3_Image_%Y%m%d_%H%M%S_UTCp1.jpg')
    times['3'].append(img3.time)

for img4 in cams[1]:
    img4.loadTimefromfilename('FE4_Image_%Y%m%d_%H%M%S_UTCp1.jpg')
    times['4'].append(img4.time)

key_pair = [(k, times['4'].index(t)) for k, t in enumerate(times['3']) if t in times['4']]
t = 0
for keys in key_pair:
    i = 0
    clouds = []
    for k in keys:
        img = cams[i][k]
        scipy.misc.imsave(
            os.path.join(plot_dir, "rectified_{0:d}_{1:d}.png".format(i, k)),
            img.image)
        image_lcn = pyclamster.Image(img)
        image_lcn.data = LCN(size=(50, 50, 3), scale=False).fit_transform(
            image_lcn.data / 256)
        w, h, _ = original_shape = image_lcn.data.shape
        raw_image_lcn = rbDetection(image_lcn.data).reshape((w * h, -1))
        label = predictor.predict(raw_image_lcn)
        label.reshape((w, h), replace=True)
        scipy.misc.imsave(
            os.path.join(plot_dir, "lables_kmean_{0:d}_{1:d}.png".format(i, k)),
            label.labels)
        masks = label.getMaskStore()
        cloud_mask_num = [1] # cloud - sky choose right number (0 or 1)
        masks.denoise(cloud_mask_num,
                      5000)
        cloud_labels_object, numLabels = masks.labelMask(cloud_mask_num)
        scipy.misc.imsave(
            os.path.join(plot_dir, "labl"
                                   "es_used_{0:d}_{1:d}.png".format(i, k)),
            cloud_labels_object.labels)
        cloud_store = cloud_labels_object.getMaskStore()
        cloud_lables = [l + 1 for l in range(numLabels)]
        clouds.append([cloud_store.getCloud(img, [k, ]) for k in cloud_lables])
        j = 0
        #print(clouds[i])
        for cloud in clouds[i]:
            scipy.misc.imsave(
                os.path.join(plot_dir, 'template_cloud_{0:d}_{1:d}_{2:d}.png'.format(i, k, j)),
                cloud.image.data)
            j += 1
        print('finished image {0:d} of camera {1:d}'.format(k, i))
        i += 1
    if not(not clouds[0] or not clouds[1]):
        matching_result, _ = matching.matching(clouds[0], clouds[1], min_match_prob=0.5)
        t = 0
        for result in matching_result:
            fig = plt.figure()
            ax = plt.subplot(1,3,1)
            ax.axis('off')
            ax.imshow(result[1].clouds[0].image.data)
            ax = plt.subplot(1,3,2)
            ax.axis('off')
            ax.imshow(result[0].prop_map, interpolation='nearest')
            ax = plt.subplot(1,3,3)
            ax.axis('off')
            ax.imshow(result[1].clouds[1].image.data)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'matching_{0:s}_{1:d}.png'.format(str(keys), t)))
            spatial_cloud = result[1]
            spatial_cloud.calc_overlapping()
            spatial_cloud.calc_position(240)
            t+=1
    # i = 0
    # for c1 in clouds[0]:
    #     j = 0
    #     for c2 in clouds[1]:
    #         result = match_template(c2.image.data, c1.image.data, pad_input=True,
    #                                 mode='reflect', constant_values=0)
    #         scipy.misc.imsave(os.path.join(plot_dir, 'cloud_matching_{0:d}_{1:d}_{2:d}.png'.format(keys[0], i, j)), result)
    #         j += 1
    #     i += 1
    print('finished image pair {0:s}'.format(str(keys)))


#
#
# for image_path in all_images:
#     img = pyclamster.image.Image(image_path)
#     img.image = img.resize((512, 512))
#
#     image.data = LCN(size=(50,50,3), scale=False).fit_transform(image.data/256)
#     raw_image = rbDetection(image.data):d}
#     w, h = original_shape = tuple(raw_image[:, :].shape)
#     raw_image = np.reshape(raw_image, (w * h, 1))
#     label = predictor.predict(raw_image)
#     label.reshape((960, 960), replace=True)
#     scipy.misc.imsave("cloud.png", label.labels)
#     masks = label.getMaskStore()
#     masks.denoise([0], 960)
#     cloud_labels, _ = masks.labelMask([0,])
#     scipy.misc.imsave("labels.png", cloud_labels.labels)
#     scipy.misc.imshow(cloud_labels.labels)
#     cloud_store = cloud_labels.getMaskStore()
#     # TODO: Here is the matching algorithm missing



