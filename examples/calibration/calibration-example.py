#!/usr/bin/env python3
import numpy as np
import pyclamster

# measured input data
row = np.array([1420,1130,1300])
col = np.array([1050,160,1420])
ele = np.array([45,80,55])
azi = np.array([170,90,-90])

lossfunction = pyclamster.calibration.CameraCalibrationLossFunction(
    known_row=row,known_col=col,known_elevation=ele,known_azimuth=azi,
    radial = lambda r:r # equidistant fisheye projection
    )

# first guess for parameters
params_firstguess = pyclamster.calibration.CameraCalibrationParameters(
    center_row = 50,
    center_col = 50,
    f = 1,
    north_angle = 0,
    alpha = 1
    )

# load an image
img = pyclamster.image.Image("../images/wettermast/Image_Wkm_Aktuell_3.jpg")

# create calibrator
calibrator = pyclamster.calibration.CameraCalibrator(img)

# let the calibrator estimate a calibration
calibration = calibrator.estimate(lossfunction, params_firstguess)

# print the results
print(calibration)
