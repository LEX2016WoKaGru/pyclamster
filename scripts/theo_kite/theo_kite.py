#!/usr/bin/env python3
import pyclamster
from pyclamster.coordinates import Coordinates3d
import numpy as np
import matplotlib.pyplot as plt

theo3_gk = Coordinates3d(x=4450909.840, y=6040800.456
    ,azimuth_offset=3/2*np.pi,azimuth_clockwise=True)
theo4_gk = Coordinates3d(x=4450713.646, y=6040934.273
    ,azimuth_offset=3/2*np.pi,azimuth_clockwise=True)
hotel_gk = Coordinates3d(x=4449525.439, y=6041088.713
    ,azimuth_offset=3/2*np.pi,azimuth_clockwise=True)

# set everything relative to theo3
theo3_gk = theo3_gk - theo3_gk
theo4_gk = theo4_gk - theo3_gk
hotel_gk = hotel_gk - theo3_gk

    
theo3_gk.plot()
theo4_gk.plot()
hotel_gk.plot()
plt.show()
