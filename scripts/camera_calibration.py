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
import os,sys,gc
import pickle
import logging
import pytz,datetime

# External modules
import matplotlib.pyplot as plt

# Internal modules
import pyclamster

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

LON=54.4947
LAT=11.2408
session = pyclamster.CameraSession(
    images="/home/yann/Studium/LEX/LEX/cam/cam3/calibration/projection/FE3*.jpg",
    longitude=LON,latitude=LAT
    )

imgsunxs = []
imgsunys = []
realsunazis = []
realsuneles = []
for image in session: # loop over all images
    # get time
    imgtime = image._get_time_from_filename(fmt="FE3_Image_%Y%m%d_%H%M%S_UTCp1.jpg")
    imgtime = pytz.utc.localize(imgtime)
    imgtime = imgtime - datetime.timedelta(hours=1)
    image.time = imgtime

    # get sun position
    imgsunpos  = image.getImageSunPosition()
    imgsunx    = imgsunpos[1]
    imgsuny    = image.data.shape[0] - imgsunpos[0] # invert y axis
    realsunazi = image.getRealSunAzimuth()
    realsunele = image.getRealSunElevation()


    # print
    logger.debug("Path: {}".format(image.path))
    logger.debug("Time: {}".format(imgtime))
    logger.debug("ImageSunPos: {}".format(imgsunpos))
    logger.debug("RealSunAzi: {}".format(realsunazi))
    logger.debug("RealSunEle: {}".format(realsunele))
    
    #plt.imshow(image.data)
    #plt.scatter(x=imgsunpos[1],y=imgsunpos[0])
    #plt.show()
    #sys.stdin.read(1) # pause

    # merge data
    imgsunxs.append(imgsunx)
    imgsunys.append(imgsuny)
    realsunazis.append(realsunazi)
    realsuneles.append(realsunele)

    del image;gc.collect() # delete and free memory
