#!/usr/bin/env python3
import time
import pyclamster
import logging
import numpy as np
import os
import pickle

logging.basicConfig(level=logging.DEBUG)

start_time = time.time()
# read an image
img = pyclamster.image.Image(os.path.join("/home/yann/Studium/LEX/LEX/cam/cam4/20160901/FE4_Image_20160901_103100_UTCp1.jpg"))
# convert to grayscale
img.image = img.convert("L")
# resize image
img.image = img.resize((300,300))


### create rectified coordinates ###
outshape=(300,300) # size of output image
rect_azimuth_offset = 3/2 * np.pi # north angle of rectified image
rect_clockwise = True
rect_x,rect_y=np.meshgrid(
    np.linspace(-20,20,num=outshape[1]),# image x coordinate goes right
    np.linspace(20,-20,num=outshape[0]) # image y coordinate goes up
    )
rect_z = 6 # rectify for height rect_z

rect_coord = pyclamster.coordinates.Coordinates3d(
    x = rect_x,
    y = rect_y,
    z = rect_z,
    azimuth_offset = rect_azimuth_offset,
    azimuth_clockwise = rect_clockwise,
    shape=outshape
    )

### create spherical coordinates of original image ###
# read calibration of wolf-3-camera
calibrationfile = "data/fe4/FE4_straightcal.pk"
calibration = pickle.load(open(calibrationfile,"rb"))

# get calibrated coordinates
img.coordinates = calibration.create_coordinates(img.data.shape)
img.coordinates.z = rect_z


### create rectification map ###
distmapfile = "data/fe4/FE4_projcalib_distmap.pk"
if True and os.path.exists(distmapfile): # use distmap from file
    logging.debug("read rectifiation map from file")
    distmap = pickle.load(open(distmapfile,"rb"))
else: # calculate distmap
    # based on regular grid
    logging.debug("calculating rectification map")
    distmap = pyclamster.fisheye.FisheyeProjection.distortionMap(
        in_coord=img.coordinates, out_coord=rect_coord, method="nearest"
        ,basedon="spherical")
    pickle.dump(distmap,open(distmapfile,"wb"))

### rectify image ##
rectimage = img.applyDistortionMap(distmap)

logging.debug("Time elapsed: {0:.3f} s".format(time.time()-start_time))

### plot results ###
import matplotlib.pyplot as plt
plt.subplot(3,4,1)
plt.title("original image (fix)")
plt.imshow(img.data, interpolation="nearest", cmap='Greys_r')
plt.subplot(3,4,2)
plt.title("image radius (calculated)")
plt.imshow(img.coordinates.radiush, interpolation="nearest")
plt.colorbar()
plt.subplot(3,4,3)
plt.title("rectified r (calculated)")
plt.imshow(rectimage.coordinates.radiush,interpolation="nearest")
plt.colorbar()
plt.subplot(3,4,4)
plt.title("rectified image (calculated)")
plt.imshow(rectimage.data, interpolation="nearest", cmap='Greys_r')
plt.subplot(3,4,5)
plt.title("image elevation (fix)")
plt.imshow(img.coordinates.elevation,interpolation="nearest")
plt.colorbar()
plt.subplot(3,4,9)
plt.title("image azimuth (fix)")
plt.imshow(img.coordinates.azimuth,interpolation="nearest")
plt.colorbar()
plt.subplot(3,4,6)
plt.title("image x (calculated)")
plt.imshow(img.coordinates.x,interpolation="nearest")
plt.colorbar()
plt.subplot(3,4,10)
plt.title("image y (calculated)")
plt.imshow(img.coordinates.y,interpolation="nearest")
plt.colorbar()
plt.subplot(3,4,7)
plt.title("rectified x (fix)")
plt.imshow(rectimage.coordinates.x,interpolation="nearest")
plt.colorbar()
plt.subplot(3,4,11)
plt.title("rectified y (fix)")
plt.imshow(rectimage.coordinates.y,interpolation="nearest")
plt.colorbar()
plt.subplot(3,4,8)
plt.title("rectified elevation (calculated)")
plt.imshow(rectimage.coordinates.elevation,interpolation="nearest")
plt.colorbar()
plt.subplot(3,4,12)
plt.title("rectified azimuth (calculated)")
plt.imshow(rectimage.coordinates.azimuth,interpolation="nearest")
plt.colorbar()
plt.show()
