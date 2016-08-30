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

# External modules

# Internal modules
import pyclamster.calibration as pyclcalib
import pyclamster.camera as pyclcam

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

cameras = {'cam3': pyclcam.Camera(),
           'cam4': pyclcam.Camera()}

with open(os.path.join(
        BASE_DIR, 'examples', 'calibration',
        'wolf-3-calibration.pickle'), 'rb') as fh:
    cali = pickle.load(fh)

# Calibrate manual the two cameras
# TODO needs a real implementation!
calibrations = {'cam3': cali,
                'cam4': cali}

# Specific the camera calibrations and pickle the camera object.
for cam in cameras:
    cameras[cam].calibration = calibrations[cam]
    with open(os.path.join(BASE_DIR, '{0:s}.pk'.format(cam)), 'wb') as fh:
        pickle.dump(cameras[cam], fh)
