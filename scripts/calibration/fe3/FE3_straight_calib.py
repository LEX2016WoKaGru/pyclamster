#!/usr/bin/env python3
import pyclamster
import numpy as np
import logging
import pickle

logging.basicConfig(level=logging.DEBUG)


imgshape = (1920,1920)

###################################################
### Read times and sun positions from filenames ###
###################################################
sun_real = pickle.load(open("data/fe3/FE3_straightcalib_sun_real.pk","rb"))
sun_img  = pickle.load(open("data/fe3/FE3_straightcalib_sun_img.pk","rb"))

proj_calib = pickle.load(open("data/FE3-projcal.pk","rb"))

#######################################
### Prepare and do the optimization ###
#######################################
# first guess for parameters
params_firstguess = pyclamster.CameraCalibrationParameters(
    960,960,
    0, # north_angle
    600 # r0
    ,100,50, 10 # r1, r2, r3
    )
# for equidistant projection: only positive r0 is sensible
params_firstguess.bounds[3]=(0,np.Inf)

# create a lossfunction
lossfunction = pyclamster.calibration.CameraCalibrationLossFunction(
    sun_img = sun_img, sun_real = sun_real,
    radial = pyclamster.FisheyePolynomialRadialFunction(params_firstguess,n=4),
    #radial = pyclamster.FisheyeEquidistantRadialFunction(params_firstguess),
    optimize_projection=True
    )

# create calibrator
calibrator = pyclamster.CameraCalibrator(shape=imgshape,method="l-bfgs-b")


# let the calibrator estimate a calibration
calibration = calibrator.estimate(lossfunction, params_firstguess)

# print the results
logging.debug("The fit: {}".format(calibration.fit))
logging.debug("The optimal parameters: {}".format(calibration.parameters))
logging.debug("The optimal residual: {}".format(calibration.lossfunc(
    calibration.parameters)))

filename = "data/fe3/FE3_straightcal.pk"
logging.debug("pickling calibration to file '{}'".format(filename))
fh = open(filename,'wb')
pickle.dump(calibration,fh)
fh.close()

cal_coords=calibration.create_coordinates()
#cal_coords.z = 1 # assume a height to see x and y

import matplotlib.pyplot as plt
plt.subplot(121)
plt.title("[calibrated]\nelevation on the image [deg]")
plt.imshow(cal_coords.elevation*360/(2*np.pi))
plt.colorbar()
plt.subplot(122)
plt.title("[calibrated]\nazimuth on the image [deg]")
plt.imshow(cal_coords.azimuth*360/(2*np.pi))
plt.colorbar()
plt.show()
