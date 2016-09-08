#!/usr/bin/env python3
import pyclamster
import numpy as np

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

print("\nCoordinates based on azimuth and radiush:")
c.shape = None
c.azimuth = np.arange(0,2*np.pi,2*np.pi/8)
c.radiush = 1
print(c)

print("\nCoordinates after parameter changing")
c.change_parameters(elevation_type="ground",keep={'elevation','azimuth','radiush'})
print(c)

print("\nCoordinates after parameter to zenith")
c.elevation_type="zenith"
c.shape = 1
c.fill(x=0,y=0)
print(c)
