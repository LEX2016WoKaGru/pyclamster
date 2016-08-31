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
import pickle
import logging

# External modules

# Internal modules
import pyclamster

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

logging.basicConfig(level=logging.DEBUG)

session = pyclamster.CameraSession("/home/yann/Studium/LEX/LEX/cam/cam3/calibration/projection/FE3*.jpg")

for image in session: # loop over all images
