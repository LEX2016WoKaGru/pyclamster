#!/usr/bin/env python3

import pyclamster
import numpy as np
import matplotlib.pyplot as plt

c = pyclamster.coordinates.Coordinates3d(
    azimuth_clockwise = False,
    azimuth_offset    = 1/2*np.pi
    )

# from x and y
print("\nCoordinates based on x and y:")
c.shape = None
c.x = [1,1,0,-1,-1,-1,0,1]
c.y = [0,1,1,1,0,-1,-1,-1]
print(c)
c.plot()
#plt.show()

alpha = np.pi/5
print("Coordinates turned by {} degrees radian".format(alpha))
c.change_parameters(azimuth_offset=alpha,keep={'radiush','azimuth'})
print(c)
#plt.figure()
c.plot()
plt.show()


