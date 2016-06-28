#!/usr/bin/env python3
import numpy as np
import logging
import pyclamster
import pysolar
import os,glob,re
import datetime,pytz

logging.basicConfig(level=logging.DEBUG)

# position of wolf-1 camera
LAT = 53.99777
LON = 9.56673

files = glob.glob("examples/images/wolf/calibration/*")
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
    r = re.compile('^Wolf_(\w+)_UTC[^_]+_sunrow(\d+)_suncol(\d+)')
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

# convert and preprocess input
sunrows = np.asarray(sunrows)
suncols = np.asarray(suncols)

azimuths   = pyclamster.deg2rad(np.asarray(azimuths))
azimuths = (azimuths + 2*np.pi) % (2*np.pi)
elevations = np.pi/2 - pyclamster.deg2rad(np.asarray(elevations))
# rotate azimuths to north
#azimuths = (pyclamster.deg2rad(azimuths) + np.pi + 2*np.pi) % (2*np.pi)
# change elevation mode from equator to zenith
#elevations = np.pi/2 - pyclamster.deg2rad(elevations) 

pixel_coords = pyclamster.Coordinates3d(
    x = suncols, y = 1920 - sunrows, # TODO: HARD CODED shape here!!!
    azimuth_clockwise = False,
    azimuth_offset=np.pi/2
    )

sun_coords = pyclamster.Coordinates3d(
    elevation = elevations, azimuth = azimuths,
    azimuth_offset = 3*np.pi/2,
    azimuth_clockwise = False
    )

# first guess for parameters
params_firstguess = pyclamster.CameraCalibrationParameters(
    center_row = 0,
    center_col = 0,
    north_angle = 0,
    r0 = 100,
    r2 = 100,
    r4 = 30,
    r6 = 0
    )


lossfunction = pyclamster.calibration.CameraCalibrationLossFunction(
    pixel_coords = pixel_coords, sun_coords = sun_coords,
    radial = pyclamster.CameraCalibrationRadialFunction(params_firstguess)
    )


# create calibrator
calibrator = pyclamster.CameraCalibrator(method="l-bfgs-b")

#raise Exception

# let the calibrator estimate a calibration
calibration = calibrator.estimate(lossfunction, params_firstguess)

# print the results
print(calibration.fit)
print(calibration.parameters)
