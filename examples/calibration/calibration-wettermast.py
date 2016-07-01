#!/usr/bin/env python3
import numpy as np
import logging
import pyclamster
import pysolar
import os,glob,re
import datetime,pytz
import pickle

logging.basicConfig(level=logging.DEBUG)

# shape of images
imgshape = (1920,1920) # hard-coded shape here...

# position of wolf-1 camera
LAT = 53.519917
LON = 10.105139

###################################################
### Read times and sun positions from filenames ###
###################################################
files = glob.glob("examples/images/wettermast/calibration/*")
files.sort()

elevations= []
azimuths  = []
sunrows   = []
suncols   = []
for imgfile in files:
    # read image
    #img = pyclamster.image.Image(imgfile)
    # read time from filename
    name = os.path.basename(imgfile)
    r = re.compile('^Wettermast_(\w+)UTC[^_]+_sunrow(\d+)_suncol(\d+)')
    m = r.match(name)
    timestr  = m.group(1)
    sunrow   = np.float(m.group(2))
    suncol   = np.float(m.group(3))
    fmt = "%Y%m%d_%H%M%S"
    time = pytz.utc.localize(datetime.datetime.strptime(timestr,fmt))
    time = time - datetime.timedelta(hours=1)
    logging.debug(time)
    ele  = pysolar.solar.get_altitude(LAT,LON,time)
    azi  = abs(pysolar.solar.get_azimuth(LAT,LON,time))
    elevations.append(ele)
    azimuths.append(azi)
    sunrows.append(sunrow)
    suncols.append(suncol)
    logging.debug("{file}: ele: {ele}, azi: {azi}, sunrow: {row}, suncol: {col}".format(
        file=imgfile,ele=ele,azi=azi,row=sunrow,col=suncol))

##############################
### Pre-process input data ###
##############################
# convert and preprocess input
sunrows = np.asarray(sunrows)
suncols = np.asarray(suncols)

# convert azimuth to radiant
azimuths   = pyclamster.deg2rad(np.asarray(azimuths))
# project azimuth on (0,2*pi)
azimuths = (azimuths + 2*np.pi) % (2*np.pi)

# convert elevation to angle from zenith 
elevations = np.pi/2 - pyclamster.deg2rad(np.asarray(elevations))


#####################################################
### Merge input data into Coordinates3d instances ###
#####################################################
# sun coordinates on the image plane based on (row, col)
sun_img = pyclamster.Coordinates3d(
    x = suncols, y = imgshape[0] - sunrows, # row increases to top
    azimuth_clockwise = False,
    azimuth_offset=0
    )
sun_img._max_print=25

# real-world astronomical sun coordiates based on (elevation,azimuth)
sun_real = pyclamster.Coordinates3d(
    elevation = elevations, azimuth = azimuths, # sun elevation and azimuth
    # astronomical azimuth increases from east to south to west --> clockwise!
    azimuth_clockwise = True,
    # astronomical azimuth is 0 in the south
    azimuth_offset = np.pi/2,
    )
sun_real._max_print=25


#######################################
### Prepare and do the optimization ###
#######################################
# first guess for parameters
params_firstguess = pyclamster.CameraCalibrationParameters(
    960, # center_row
    960, # center_col
    0, # north_angle
    600, # r0
    100, # r0
    50, # r0
    10, # r0
    )
# for equidistant projection: only positive r0 is sensible
params_firstguess.bounds[3]=(0,np.Inf)

# create a lossfunction
lossfunction = pyclamster.calibration.CameraCalibrationLossFunction(
    sun_img = sun_img, sun_real = sun_real,
    radial = pyclamster.FisheyePolynomialRadialFunction(params_firstguess,n=4)
    )

# create calibrator
calibrator = pyclamster.CameraCalibrator(shape=imgshape,method="l-bfgs-b")


# let the calibrator estimate a calibration
calibration = calibrator.estimate(lossfunction, params_firstguess)

# print the results
logging.debug(calibration.fit)
logging.debug(calibration.parameters)

filename = "examples/calibration/wettermst-calibration.pk"
logging.debug("pickling calibration to file '{}'".format(filename))
fh = open(filename,'wb')
pickle.dump(calibration,fh)
fh.close()

cal_coords=calibration.create_coordinates()
cal_coords.z = 1 # assume a height to see x and y

import matplotlib.pyplot as plt
plt.subplot(221)
plt.title("[calibrated]\nelevation on the image [deg]")
plt.imshow(cal_coords.elevation*360/(2*np.pi))
plt.colorbar()
plt.subplot(222)
plt.title("[calibrated]\nazimuth on the image [deg]")
plt.imshow(cal_coords.azimuth*360/(2*np.pi))
plt.colorbar()
plt.subplot(223)
plt.title("[calibrated]\n[z=1 plane]\nreal-world x on the image [m]")
plt.imshow(cal_coords.x)
plt.colorbar()
plt.subplot(224)
plt.title("[calibrated]\n[z=1 plane]\nreal-world y on the image [m]")
plt.imshow(cal_coords.y)
plt.colorbar()
plt.show()
