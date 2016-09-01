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

# External modules
import numpy as np
import scipy.misc

# Internal modules
import pyclamster
from pyclamster.clustering.preprocess import LCN
from pyclamster.functions import rbDetection

warnings.catch_warnings()
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

image_directory = os.path.join(BASE_DIR, "examples", "images", "lex")
trained_models = os.path.join(BASE_DIR, "data")
plot_dir = os.path.join(BASE_DIR, 'plots')

predictor = pickle.load(open(os.path.join(trained_models, "kmeans.pk"), "rb"))
fisheye = pyclamster.fisheye.FisheyeProjection("equidistant")
### create rectified coordinates ###
outshape=(256,256)
rect_azimuth_offset = np.pi / 2 # north angle of rectified image
rect_clockwise = False
rect_x,rect_y=np.meshgrid(
    np.linspace(-20,20,num=outshape[1]),# image x coordinate goes right
    np.linspace(20,-20,num=outshape[0]) # image y coordinate goes up
    )
rect_z = 15 # rectify for height rect_z

rect_coord = pyclamster.coordinates.Coordinates3d(
    x = rect_x,
    y = rect_y,
    z = rect_z,
    azimuth_offset = rect_azimuth_offset,
    azimuth_clockwise = rect_clockwise,
    shape=outshape
    )

### create spherical coordinates of original image ###



cams = []
cams.append(pyclamster.CameraSession(images=os.path.join(image_directory, 'cam3', 'FE3*.jpg'), longitude=11.240817, latitude=54.4947))
cams.append(pyclamster.CameraSession(images=os.path.join(image_directory, 'cam4', 'FE4*.jpg'), longitude=11.237684, latitude=54.49587))

times = {'3': [], '4': []}
# load times
for img3 in cams[0]:
    img3.loadTimefromfilename('FE3_Image_%Y%m%d_%H%M%S_UTCp1.jpg')
    times['3'].append(img3.time)

for img4 in cams[1]:
    img4.loadTimefromfilename('FE4_Image_%Y%m%d_%H%M%S_UTCp1.jpg')
    times['4'].append(img4.time)

key_pair = [(k, times['4'].index(t)) for k, t in enumerate(times['3']) if t in times['4']]
distmap = []
for keys in key_pair:
    i = 0
    for k in keys:
        img = cams[i][k]
        img.resize((512, 512))
        shape = np.shape(img.data)[:2]  # shape of image
        image_north_angle = 6 * np.pi / 5  # north angle ON the original image
        orig_azimuth_offset = np.pi / 2  # "north angle" on image coordinates
        center = None  # center of elevation/azimuth in the image
        maxelepos = (
        0, int(shape[1] / 2))  # (one) position of maxium elevation
        maxele = np.pi / 2.2  # maximum elevation on the image border, < 90° here
        img.coordinates.azimuth_offset = orig_azimuth_offset
        img.coordinates.azimuth_clockwise = False
        img.coordinates.elevation = fisheye.createFisheyeElevation(
            shape,
            maxelepos=maxelepos,
            maxele=maxele,
            center=center)
        img.coordinates.azimuth = fisheye.createAzimuth(
            shape,
            maxelepos=maxelepos,
            center=center,
            north_angle=image_north_angle,
            clockwise=False
        )
        img.coordinates.z = rect_z
        try:
            img = img.applyDistortionMap(distmap[i])
        except:
            distmap.append(fisheye.distortionMap(in_coord=img.coordinates,
                                      out_coord=rect_coord, method="nearest"))
            img = img.applyDistortionMap(distmap[i])
        scipy.misc.imsave(
            os.path.join(plot_dir, "rectified_{0:d}_{1:d}.png".format(i, k)),
            label.labels)
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


        i += 1

#
#
# for image_path in all_images:
#     img = pyclamster.image.Image(image_path)
#     img.image = img.resize((512, 512))
#
#     image.data = LCN(size=(50,50,3), scale=False).fit_transform(image.data/256)
#     raw_image = rbDetection(image.data)
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



