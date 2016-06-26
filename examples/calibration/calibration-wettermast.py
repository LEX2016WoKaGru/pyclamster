#!/usr/bin/env python3
import numpy as np
import logging
import pyclamster
import pysolar
import os,glob,re
import datetime,pytz

logging.basicConfig(level=logging.DEBUG)

# position of wettermast camera
LAT = 53.519917
LON = 10.105139

files = glob.glob("/home/yann/Bilder/Wettermast/*")

elevations= []
azimuths  = []
sunrows   = []
suncols   = []
for imgfile in files:
    # read image
    img = pyclamster.image.Image(imgfile)
    # read time from filename
    name = os.path.basename(imgfile)
    r = re.compile('^Wettermast_(\w+)UTC[^_]+_sunrow(\d+)_suncol(\d+)')
    m = r.match(name)
    timestr  = m.group(1)
    sunrow   = np.float(m.group(2))
    suncol   = np.float(m.group(3))
    fmt = "%Y%m%d_%H%M%S"
    time = datetime.datetime.strptime(timestr,fmt)
    ele  = pysolar.solar.get_altitude(LAT,LON,time)
    azi  = pysolar.solar.get_azimuth(LAT,LON,time)
    elevations.append(ele)
    azimuths.append(azi)
    sunrows.append(sunrow)
    suncols.append(suncol)
    logging.debug("{file}: ele: {ele}, azi: {azi}, sunrow: {row}, suncol: {col}".format(
        file=imgfile,ele=ele,azi=azi,row=sunrow,col=suncol))

# convert and preprocess input
azimuths = np.asarray(azimuths)
elevations = np.asarray(elevations)
azimuths = (azimuths / 360 * (2*np.pi) + np.pi + 2*np.pi) % (2*np.pi)
elevations = np.pi/2 - elevations / 360 * (2*np.pi)

center = pyclamster.coordinates.CarthesianCoordinates3d(x=0,y=0,z=0)
pixel_coords = pyclamster.coordinates.CarthesianCoordinates3d(
    x = suncols, y = sunrows,
    center = center
    )

sun_coords = pyclamster.coordinates.SphericalCoordinates3d(
    elevation = elevations, azimuth = azimuths
    )

lossfunction = pyclamster.calibration.CameraCalibrationLossFunction(
    pixel_coords = pixel_coords, sun_coords = sun_coords,
    radial = lambda r:r # equidistant fisheye projection
    )

# first guess for parameters
params_firstguess = pyclamster.calibration.CameraCalibrationParameters(
    center_row = 960,
    center_col = 960,
    f = 1,
    north_angle = 0,
    alpha = 1
    )

# create calibrator
calibrator = pyclamster.calibration.CameraCalibrator(img,method="l-bfgs-b")

# let the calibrator estimate a calibration
calibration = calibrator.estimate(lossfunction, params_firstguess)

# print the results
print(calibration.fit)
print(calibration.parameters)
