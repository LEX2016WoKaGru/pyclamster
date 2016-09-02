#!/usr/bin/env python3
import pyclamster
import logging
import pickle
import os,sys

# set up logging
logging.basicConfig(level=logging.DEBUG)

sessionfile = "data/sessions/FE4_session.pk"
try: # maybe there is already a session
    session = pickle.load(open(sessionfile,"rb"))
except: # if not
    # read calibration
    calib = pickle.load(open("data/fe4/FE4_straightcal.pk","rb"))
    
    # create session
    session = pyclamster.CameraSession(
        longitude  = 54.49587,
        latitude   = 11.237683,
        imgshape   = (1920,1920),
        smallshape = (300,300),
        rectshape  = (200,200),
        calibration = calib
        )
    # add images to session
    session.add_images("/home/yann/Studium/LEX/LEX/cam/cam4/20160901/FE4_Image_20160901_09*.jpg")
    
    # create distortion map
    session.createDistortionMap()

    # save thie session
    session.save(sessionfile)

# loop over all images
for image in session.iterate_over_rectified_images():
    image.show()
    sys.stdin.read(1)
