#!/usr/bin/env python3
import logging
import gc
import sys
logging.basicConfig(level=logging.DEBUG)

import pyclamster
session = pyclamster.CameraSession("/home/yann/Studium/LEX/LEX/cam/cam3/FE3*.jpg")

for image in session:
    image.show()
    sys.stdin.read(1)
    del image
    gc.collect()
