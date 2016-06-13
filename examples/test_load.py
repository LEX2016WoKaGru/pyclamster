# -*- coding: utf-8 -*-
"""
Created on 10.06.16

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

# External modules
import numpy as np
import tables

# Internal modules
from pyclamster.clustering import preprocess

__version__ = ""


hdf5_save_path = "/home/tfinn/Projects/pyclamster/examples/training.hdf5"

compression_filter = tables.Filters(complevel=5, complib='blosc')
hdf5_file = tables.open_file(hdf5_save_path, filters=compression_filter,
                             mode="r", title="Trainings data")

X = hdf5_file.root.wettermast.patches[:]/255
y = hdf5_file.root.wettermast.labels[:]

hdf5_file.close()

for x in np.nditer(X, op_flags=['readwrite']):
    x[...] = preprocess.LCN(patches=True, copy=True).fit_transform(x)

#X = [preprocess.LCN(patches=True).fit_transform(X[i]) for i in range(X.shape[0])]
X = preprocess.ZCA(bias=0).fit_transform(X)

