#!/usr/bin/env python3
import pyclamster
from pyclamster.coordinates import Coordinates3d
import numpy as np
import matplotlib.pyplot as plt
import logging
import pickle

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

session3 = pickle.load(open('data/sessions/FE3_session_new.pk','rb'))
session4 = pickle.load(open('data/sessions/FE4_session_new.pk','rb'))

cam3,time3=pickle.load(open("data/cam_kite/FE3_cam_kite_new.pk","rb"))
cam4,time4=pickle.load(open("data/cam_kite/FE4_cam_kite_new.pk","rb"))


# calculate 3d positions via doppelanschnitt
doppel = pyclamster.doppelanschnitt_Coordinates3d(
    aziele1 = cam3, aziele2 = cam4,
    pos1 = session4.position, pos2 = session3.position
    )


# plot results
ax = doppel.plot3d(method="line")
plt.show()
