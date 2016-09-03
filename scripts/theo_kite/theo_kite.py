#!/usr/bin/env python3
import pyclamster
from pyclamster.coordinates import Coordinates3d
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG)


def read_theo_data(file_list):
    m = []
    for measure_files in file_list:
        time1 = []
        star1 = []
        azim1 = []
        elev1 = []
        
        for k in range(len(measure_files)):
            time1.append([])
            star1.append([])
            azim1.append([])
            elev1.append([])
            with open(measure_files[k]) as f:
                f.readline()
                eof = False
                while not eof:
                    line = f.readline()
                    if line[0]=='E':
                        eof = True
                    time1[-1].append(int(line[3:6]))
                    star1[-1].append(1 if line[6:7]=='*' else 0)
                    azim1[-1].append(pyclamster.utils.deg2rad(float(line[7:13])))
                    elev1[-1].append(pyclamster.utils.deg2rad(float(line[15:20])))
        time = []
        star = []
        azim = []
        elev = []
        
        for i in range(len(measure_files)):
            time.append(np.array(time1[i]))
            star.append(np.array(star1[i]))
            azim.append(np.array(azim1[i]))
            elev.append(np.array(elev1[i]))
    
        maxlen = min([len(time[i]) for i in [0,1]])
        m1 = np.empty([6,maxlen])
        m1[0,:] = time[0][:maxlen]
        m1[1,:] = star[0][:maxlen]+star[1][:maxlen]
        m1[2,:] = azim[0][:maxlen]
        m1[3,:] = azim[1][:maxlen]
        m1[4,:] = elev[0][:maxlen]
        m1[5,:] = elev[1][:maxlen]
        m.append(m1)
    return m


#t3m1,t4m1,t3m2,t4m2
datafiles = [["scripts/theo_kite/Theo_daten/rot/000R_20160901_110235.td4",
              "scripts/theo_kite/Theo_daten/gelb/000G_20160901_110235.td4"],
             ["scripts/theo_kite/Theo_daten/rot/000R_20160901_112343.td4",
              "scripts/theo_kite/Theo_daten/gelb/000G_20160901_112344.td4"]]

data = read_theo_data(datafiles)

theo3_gk = Coordinates3d(x=4450909.840, y=6040800.456, z=6
    ,azimuth_offset=3/2*np.pi,azimuth_clockwise=True)
theo4_gk = Coordinates3d(x=4450713.646, y=6040934.273, z=1
    ,azimuth_offset=3/2*np.pi,azimuth_clockwise=True)
hotel_gk = Coordinates3d(x=4449525.439, y=6041088.713
    ,azimuth_offset=3/2*np.pi,azimuth_clockwise=True)

nordcorr3 = -np.arctan2(hotel_gk.y-theo3_gk.y,hotel_gk.x-theo3_gk.x)+np.pi*0.5
nordcorr4 = -np.arctan2(hotel_gk.y-theo4_gk.y,hotel_gk.x-theo4_gk.x)+np.pi*0.5

# first measurements
theo3_1 = Coordinates3d(
    azimuth   = pyclamster.pos_rad(data[0][2,:]+nordcorr3),
    elevation = data[0][4,:],
    azimuth_clockwise = True,
    azimuth_offset = 3/2*np.pi,
    elevation_type = "ground"
    )

theo4_1 = Coordinates3d(
    azimuth   = pyclamster.pos_rad(data[0][3,:]+nordcorr4),
    elevation = data[0][5,:],
    azimuth_clockwise = True,
    azimuth_offset = 3/2*np.pi,
    elevation_type = "ground"
    )

# second measurements
theo3_2 = Coordinates3d(
    azimuth   = pyclamster.pos_rad(data[1][2,:]+nordcorr3),
    elevation = data[1][4,:],
    azimuth_clockwise = True,
    azimuth_offset = 3/2*np.pi,
    elevation_type = "ground"
    )

theo4_2 = Coordinates3d(
    azimuth   = pyclamster.pos_rad(data[1][3,:]+nordcorr4),
    elevation = data[1][5,:],
    azimuth_clockwise = True,
    azimuth_offset = 3/2*np.pi,
    elevation_type = "ground"
    )

# calculate 3d positions via doppelanschnitt
doppel1 = pyclamster.doppelanschnitt_Coordinates3d(
    aziele1 = theo3_1, aziele2 = theo4_1,
    pos1 = theo3_gk, pos2 = theo4_gk
    )

doppel2 = pyclamster.doppelanschnitt_Coordinates3d(
    aziele1 = theo3_2, aziele2 = theo4_2,
    pos1 = theo3_gk, pos2 = theo4_gk
    )

# plot results
ax = doppel1.plot3d(method="line")
ax.set_zlim(0,300)
ax = doppel2.plot3d(method="line")
ax.set_zlim(0,300)
plt.show()
