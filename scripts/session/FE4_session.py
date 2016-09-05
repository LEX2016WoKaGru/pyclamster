#!/usr/bin/env python3
import pyclamster
import logging
import pickle
import os,sys

# set up logging
logging.basicConfig(level=logging.DEBUG)

sessionfile = "data/sessions/FE4_session_new.pk"
try: # maybe there is already a session
    session = pickle.load(open(sessionfile,"rb"))
except: # if not
    # read calibration
    calib = pickle.load(open("data/fe4/FE4_straight+projcal.pk","rb"))
    
    # create session
    session = pyclamster.CameraSession(
        longitude  = 54.49587,
        latitude   = 11.237683,
        heightNN   = 0,
        zone       = 32,
        imgshape   = (1920,1920),
        smallshape = (500,500),
        rectshape  = (300,300),
        calibration = calib
        )
    # add images to session
    session.add_images("/home/yann/Studium/LEX/LEX/cam/cam4/20160901/*.jpg")
    
    # create distortion map
    session.createDistortionMap(max_angle=pyclamster.deg2rad(75))

    # save thie session
    session.reset_images()
    session.save(sessionfile)

# loop over all images
#for image in session.iterate_over_rectified_images():
    #filename = image._get_time_from_filename("FE4_Image_%Y%m%d_%H%M%S_UTCp1.jpg")
    #image.image.save("plots/images/fe4/{}-rect.jpg".format(filename))
