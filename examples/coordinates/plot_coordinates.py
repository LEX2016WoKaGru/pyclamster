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
#c.plot()
#plt.show()

for angle in np.linspace(0,2*np.pi,10):
    print("Coordinates turned by {} degrees radian".format(angle))
    # turn by changing the azimuth_offset
    c.change_parameters(azimuth_offset=angle,keep={'radiush','azimuth'})
    # turn by changing azimuth
    #c.fill(radiush=c.radiush, azimuth=(c.azimuth+np.pi/5)%(2*np.pi))
    print(c)
    c.plot()

plt.show()

