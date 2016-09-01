#!/usr/bin/env python3
import os
import time
import pyclamster
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
image_path = os.path.join(BASE_DIR, 'examples', 'images', 'wettermast',
                          'Image_Wkm_Aktuell_2.jpg')

start_time = time.time()
# read an image
img = pyclamster.image.Image(image_path)
# convert to grayscale
img.image = img.convert("L")
# resize image
img.image = img.resize((200,200))

### create a fisheye projection object ###
f=pyclamster.fisheye.FisheyeProjection("equidistant")

### create rectified coordinates ###
outshape=(50,50) # size of output image
rect_azimuth_offset = np.pi / 2 # north angle of rectified image
rect_clockwise = False
rect_x,rect_y=np.meshgrid(
    np.linspace(-20,20,num=outshape[1]),# image x coordinate goes right
    np.linspace(20,-20,num=outshape[0]) # image y coordinate goes up
    )
rect_z = 15 # rectify for height rect_z

rect_coord = pyclamster.coordinates.Coordinates3d(
    x = rect_x,
    y = rect_y,
    z = rect_z,
    azimuth_offset = rect_azimuth_offset,
    azimuth_clockwise = rect_clockwise,
    shape=outshape
    )

### create spherical coordinates of original image ###
shape=np.shape(img.data)[:2] # shape of image
image_north_angle = 6 * np.pi / 5 # north angle ON the original image
orig_azimuth_offset = np.pi / 2 # "north angle" on image coordinates
center = None # center of elevation/azimuth in the image
maxelepos = (0,int(shape[1]/2)) # (one) position of maxium elevation
maxele = np.pi / 2.2 # maximum elevation on the image border, < 90° here

img.coordinates.azimuth_offset = orig_azimuth_offset
img.coordinates.azimuth_clockwise = False

logging.debug("setting image elevation")
img.coordinates.elevation = f.createFisheyeElevation(
    shape,
    maxelepos=maxelepos,
    maxele=maxele,
    center=center
    )
logging.debug("mean image elevation is {}".format(img.coordinates.elevation.mean()))

logging.debug("setting image azimuth")
img.coordinates.azimuth = f.createAzimuth(
    shape,
    maxelepos=maxelepos,
    center=center,
    north_angle = image_north_angle,
    clockwise=False
    )

logging.debug("setting image radius")
img.coordinates.z = rect_z

### create rectification map ###
# based on regular grid
logging.debug("calculating rectification map")
distmap = f.distortionMap(in_coord=img.coordinates, 
    out_coord=rect_coord, method="nearest")

### rectify image ##
rectimage = img.applyDistortionMap(distmap)

### plot results ###
import matplotlib.pyplot as plt
plt.subplot(3,4,1)
plt.title("original image (fix)")
plt.imshow(img.data, interpolation="nearest")
plt.subplot(3,4,2)
plt.title("image radius (calculated)")
plt.imshow(img.coordinates.radius, interpolation="nearest")
plt.colorbar()
plt.subplot(3,4,3)
plt.title("rectified r (calculated)")
plt.imshow(rect_coord.radius,interpolation="nearest")
plt.colorbar()
plt.subplot(3,4,4)
plt.title("rectified image (calculated)")
plt.imshow(rectimage.data, interpolation="nearest")
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
plt.imshow(rect_coord.x,interpolation="nearest")
plt.colorbar()
plt.subplot(3,4,11)
plt.title("rectified y (fix)")
plt.imshow(rect_coord.y,interpolation="nearest")
plt.colorbar()
plt.subplot(3,4,8)
plt.title("rectified elevation (calculated)")
plt.imshow(rect_coord.elevation,interpolation="nearest")
plt.colorbar()
plt.subplot(3,4,12)
plt.title("rectified azimuth (calculated)")
plt.imshow(rect_coord.azimuth,interpolation="nearest")
plt.colorbar()
logging.debug("Time elapsed: {0:.3f} s".format(time.time()-start_time))
plt.show()
