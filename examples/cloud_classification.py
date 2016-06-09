# -*- coding: utf-8 -*-
"""
Created on 06.06.16

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
import pickle

# External modules
import scipy.misc
import tables

# Internal modules
from pyextremelm import ELMRegressor

__version__ = ""

hdf5_save_path = "./training.hdf5"

compression_filter = tables.Filters(complevel=5, complib='blosc')
hdf5_file = tables.open_file(hdf5_save_path, filters=compression_filter,
                             mode="r", title="Trainings data")

#index = hdf5_file.root.wettermast.image_storage[0]
index = 420

img = hdf5_file.root.wettermast.patches[index]
print(index, hdf5_file.root.wettermast.labels[index])
scipy.misc.imsave("test.png", img)


hdf5_file.close()