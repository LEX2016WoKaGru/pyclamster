#!/usr/bin/env python3
import numpy as np
import logging
import pyclamster

logging.basicConfig(level=logging.DEBUG)

# load an image
img = pyclamster.image.Image("../images/wettermast/Image_Wkm_Aktuell_3.jpg")

# measured input data
row = np.array([1420,1130,1300])
col = np.array([1050,160,1420])
ele = np.array([45,80,55])
azi = np.array([170,90,-90])

pixel_coords = pyclamster.coordinates.CarthesianCoordinates3d(
    x = col, y = row
    )

sun_coords = pyclamster.coordinates.SphericalCoordinates3d(
    elevation = ele, azimuth = azi
    )

lossfunction = pyclamster.calibration.CameraCalibrationLossFunction(
    pixel_coords = pixel_coords, sun_coords = sun_coords,
    radial = lambda r:r # equidistant fisheye projection
    )

# first guess for parameters
params_firstguess = pyclamster.calibration.CameraCalibrationParameters(
    center_row = img.data.shape[0]/2,
    center_col = img.data.shape[1]/2,
    f = 1,
    north_angle = 0,
    alpha = 1
    )

# create calibrator
calibrator = pyclamster.calibration.CameraCalibrator(img,method="L-BFGS-B")

# let the calibrator estimate a calibration
calibration = calibrator.estimate(lossfunction, params_firstguess)

# print the results
print(calibration)
