#!/usr/bin/env python3
import pyclamster
import numpy as np
import logging
import pickle

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

imgshape = (1920,1920)

###################################################
### Read times and sun positions from filenames ###
###################################################
sun_real = pickle.load(open("data/fe3/FE3_projcalib_sun_real.pk","rb"))
sun_img  = pickle.load(open("data/fe3/FE3_projcalib_sun_img.pk","rb"))

sun_img.fill( 
    x=sun_img.x-imgshape[1]/2,
    y=sun_img.y-imgshape[0]/2
    )

#######################################
### Prepare and do the optimization ###
#######################################
# first guess for parameters
params_firstguess = pyclamster.CameraCalibrationParameters(
    0,0,0, # assume no rotation (perfect northing/zenithing)
    600 # r1
    ,0,0, 0  # r2, r3, r4
    )
# for equidistant projection: only positive r0 is sensible
params_firstguess.bounds[3]=(0,np.Inf)

# create a lossfunction
lossfunction = pyclamster.calibration.CameraCalibrationLossFunction(
    sun_img = sun_img, sun_real = sun_real,
    radial = pyclamster.FisheyePolynomialRadialFunction(params_firstguess,n=4),
    #radial = pyclamster.FisheyeEquidistantRadialFunction(params_firstguess),
    imgshape = imgshape,
    optimize_projection=True
    )
lossfunction.PLOT     = False
lossfunction.SKIPPLOT = 50
lossfunction.VERBOSE  = True

# create calibrator
calibrator = pyclamster.CameraCalibrator(shape=imgshape,method="l-bfgs-b")


# let the calibrator estimate a calibration
calibration = calibrator.estimate(lossfunction, params_firstguess)

# print the results
logging.debug("The fit: {}".format(calibration.fit))
logging.debug("The optimal parameters: {}".format(calibration.parameters))
logging.debug("The optimal residual: {}".format(calibration.lossfunc(
    calibration.parameters)))

filename = "data/fe3/FE3_projcal.pk"
logging.debug("pickling calibration to file '{}'".format(filename))
try:
    fh = open(filename,'wb')
    # no need to save whole set of positions
    calibration.lossfunc.sun_img.shape=None
    calibration.lossfunc.sun_real.shape=None
    pickle.dump(calibration,fh)
    fh.close()
except:
    logger.error("something went wrong with pickling...")

cal_coords=calibration.create_coordinates((300,300))

import matplotlib.pyplot as plt
plt.subplot(121)
plt.title("[calibrated]\nelevation on the image [deg]")
plt.imshow(cal_coords.elevation*360/(2*np.pi),cmap="Blues")
plt.colorbar()
plt.subplot(122)
plt.title("[calibrated]\nazimuth on the image [deg]")
plt.imshow(cal_coords.azimuth*360/(2*np.pi),cmap="Blues")
plt.colorbar()
plt.show()
