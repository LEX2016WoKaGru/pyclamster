#!/usr/bin/env python3
import pyclamster
import numpy as np
import matplotlib.pyplot as plt

theo1 = pyclamster.Coordinates3d(
    azimuth =   [0,3],
    elevation = [pyclamster.deg2rad(45),np.nan],
    azimuth_clockwise = False,
    azimuth_offset = 0,
    elevation_type = "zenith"
    )

pos1 = pyclamster.Coordinates3d(
    x = 0, y = 0, z = 0
    )


theo2 = pyclamster.Coordinates3d(
    azimuth =   [np.pi,np.nan],
    elevation = [pyclamster.deg2rad(45),1],
    azimuth_clockwise = True,
    azimuth_offset = 0,
    elevation_type = "ground"
    )

pos2 = pyclamster.Coordinates3d(
    x = 1, y = 1, z = 0
    )
    
doppel = pyclamster.positioning.doppelanschnitt_Coordinates3d(
    aziele1 = theo1,
    aziele2 = theo2,
    pos1    = pos1,
    pos2    = pos2
    )

ax = doppel.plot3d()
ax.set_xlim(0,1)
ax.set_ylim(-1,1)
ax.set_zlim(0,1)
ax.scatter3D(pos1.x,pos1.y,pos1.z,color='r')
ax.scatter3D(pos2.x,pos2.y,pos2.z,color='g')
plt.show()
