# -*- coding: utf-8 -*-
"""
Created on 15.07.16

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
import zipfile
import sys
import glob
import datetime

# External modules
import scipy.misc
import numpy as np
import tables

# Internal modules


__version__ = "0.1"


def get_camera_day(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    parted_file = filename.split('_')
    return parted_file[1], parted_file[2]


def get_image_array(path, hdf_file, grp):
    zf = zipfile.ZipFile(path, 'r')
    image_store = None
    date_store = None
    i = 0
    for file in zf.infolist():
        image = scipy.misc.imread(zf.open(file), mode='RGB')
        img_shp = image.shape
        image = image.reshape((1, img_shp[0], img_shp[1], img_shp[2]))
        date = file.filename
        date = datetime.datetime.strptime(date, 'Image_%Y%m%d_%H%M%S_%Zp1.jpg')
        date = np.array(date.timestamp()).reshape((1,1))
        if (image_store is None) or (date_store is None):
            if grp.__contains__('images'):
                image_store = grp._f_get_child('images')
            else:
                image_store = hdf_file.createEArray(
                    grp, 'images', tables.Atom.from_dtype(image.dtype),
                    shape=(0, img_shp[0], img_shp[1], img_shp[2]))
            if grp.__contains__('dates'):
                date_store = grp._f_get_child('dates')
            else:
                date_store = hdf_file.createEArray(
                    grp, 'dates', tables.Atom.from_dtype(date.dtype),
                    shape=(0, 1))
        if not date in date_store:
            date_store.append(date)
            image_store.append(image)
        i += 1
        print('fininshed {0:d}/{1:d}'.format(i, len(zf.infolist())))
    zf.close()
    print('Converted images into hdf5 for file {:s}'.format(path))


def main(zip_path, hdf_path):
    if os.path.isdir(zip_path):
        zip_path = os.path.join(zip_path, "*.zip")
    zip_files = sorted(glob.glob(zip_path))
    compression_filter = tables.Filters(complevel=5, complib='blosc')
    hdf5_file = tables.open_file(hdf_path, filters=compression_filter,
                                 mode="a", title="Camera images")
    for file in zip_files:
        print('Opened {:s}'.format(file))
        cam, day = get_camera_day(file)
        if not cam in hdf5_file.get_node('/'):
            grp_cam = hdf5_file.create_group(
                "/", cam, 'Camera: {0:s}'.format(cam))
        else:
            grp_cam = hdf5_file.get_node('/', cam)
        get_image_array(file, hdf5_file, grp_cam)
    hdf5_file.close()
    print('Finished writing {0:s} to {1:s}'.format(zip_path, hdf_path))

if __name__ == '__main__':
    if len(sys.argv)>2:
        zip_path = sys.argv[1]
        hdf_path = sys.argv[2]
    else:
        zip_path = input("Please insert where the zip files could be found:\n")
        hdf_path = input("Please insert where the hdf5 file should be saved:\n")
    main(zip_path, hdf_path)
