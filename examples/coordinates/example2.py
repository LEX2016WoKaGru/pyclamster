#!/usr/bin/env python3
import pyclamster
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

c = pyclamster.Coordinates3d()

c.shape = (5,)
c.z = 1
c.radiush = 1
c.azimuth = np.linspace(0,2*np.pi,5)

print(c)
