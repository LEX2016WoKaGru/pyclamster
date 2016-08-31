# -*- coding: utf-8 -*-
"""
Created on 29.08.16

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
import glob
import datetime
import pickle

# External modules
import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import skimage.morphology

# Internal modules
import pyclamster.image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
r_thres = 220
image_dir = '/home/tfinn/Data/Cloud_camera/lex/cam3/calibration/projection/'

image_path_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
print(image_path_list)
sun_positions = {}

for img_path in image_path_list:
    img = pyclamster.image.Image(img_path)
    img.loadTimefromfilename('FE3_Image_%Y%m%d_%H%M%S_UTCp1.jpg')
    img.time = img.time-datetime.timedelta(hours=1)
    img.data = scipy.ndimage.filters.gaussian_filter(img.data, 3)
    img.data[:100, :, :] = 0
    img.data[-100:, :, :] = 0
    r_ch = img.data[:,:,0]
    sun_filter = r_ch > r_thres
    sun_filter = skimage.morphology.remove_small_objects(sun_filter, 7)
    sun_position = scipy.ndimage.center_of_mass(sun_filter)
    sun_position = [sun_position[1], sun_position[0]]
    print('{0:s} finished'.format(os.path.basename(img_path)))

pickle.dump(sun_positions, open('sun_positions_cam_3.pk', mode='wb'))
