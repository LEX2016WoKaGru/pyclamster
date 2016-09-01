import matplotlib.pyplot as plt
import numpy as np
import pyclamster
from pyclamster.coordinates import Coordinates3d


theo3 = np.array([4450909.840, 6040800.456])
theo4 = np.array([4450713.646, 6040934.273])
hotel = np.array([4449525.439, 6041088.713]) 

#np.arctan2(y,x)

nordcorr3 = np.arctan2(hotel[1]-theo3[1],hotel[0]-theo3[0])*180/np.pi-90
nordcorr4 = np.arctan2(hotel[1]-theo4[1],hotel[0]-theo4[0])*180/np.pi-90

#print(nordcorr3)
#print(nordcorr4)
#az+nordcorrectur

if 0:
    plt.plot(0,0,'bo')
    plt.plot(theo4[0]-theo3[0],theo4[1]-theo3[1],'go')
    plt.plot(hotel[0]-theo3[0],hotel[1]-theo3[1],'rx')
    plt.xlim([-1400,0])
    plt.ylim([0,1400])
    plt.show()

datafiles = {'t3m1':"./Theo_daten/rot/000R_20160901_110235.td4",
        't3m2':"./Theo_daten/rot/000R_20160901_112343.td4",
        't4m1':"./Theo_daten/gelb/000G_20160901_110235.td4",
        't4m2':"./Theo_daten/gelb/000G_20160901_112344.td4"}


time1 = []
star1 = []
azim1 = []
elev1 = []

for i,k in enumerate(datafiles.keys()):
    time1.append([])
    star1.append([])
    azim1.append([])
    elev1.append([])
    with open(datafiles[k]) as f:
        f.readline()
        eof = False
        while not eof:
            line = f.readline()
            if line[0]=='E':
                eof = True
            time1[-1].append(int(line[3:6]))
            star1[-1].append(1 if line[6:7]=='*' else 0)
            azim1[-1].append(float(line[7:13]))
            elev1[-1].append(float(line[15:20]))
time = []
star = []
azim = []
elev = []

for i in range(len(datafiles.keys())):
    time.append(np.array(time1[i]))
    star.append(np.array(star1[i]))
    azim.append(np.array(azim1[i]))
    elev.append(np.array(elev1[i]))

maxlen = min([len(time[i]) for i in [0,2]])
m1 = np.empty([6,maxlen])
m1[0,:] = time[0][:maxlen]
m1[1,:] = star[0][:maxlen]+star[2][:maxlen]
m1[2,:] = azim[0][:maxlen]+nordcorr3
m1[3,:] = azim[2][:maxlen]+nordcorr4
m1[4,:] = elev[0][:maxlen]
m1[5,:] = elev[2][:maxlen]

m1_theo3 = Coordinates3d(azimuth=pyclamster.deg2rad(m1[2,]),
    elevation=pyclamster.deg2rad(m1[4,]))
m1_theo4 = Coordinates3d(azimuth=pyclamster.deg2rad(m1[3,]),
    elevation=pyclamster.deg2rad(m1[5,]))

maxlen = min([len(time[i]) for i in [1,3]])
m2 = np.empty([6,maxlen])
m2[0,:] = time[1][:maxlen]
m2[1,:] = star[1][:maxlen]+star[3][:maxlen]
m2[2,:] = azim[1][:maxlen]+nordcorr3
m2[3,:] = azim[3][:maxlen]+nordcorr4
m2[4,:] = elev[1][:maxlen]
m2[5,:] = elev[3][:maxlen]

m2_theo3 = Coordinates3d(azimuth=pyclamster.deg2rad(m2[2,]),
    elevation=pyclamster.deg2rad(m2[4,]))
m2_theo4 = Coordinates3d(azimuth=pyclamster.deg2rad(m2[5,]),
    elevation=pyclamster.deg2rad(m2[3,]))
