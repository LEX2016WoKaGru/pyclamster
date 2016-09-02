import matplotlib.pyplot as plt
import sys
import numpy as np
import pyclamster
from pyclamster.coordinates import Coordinates3d


theo3 = np.array([4450909.840, 6040800.456,6])#6
theo4 = np.array([4450713.646, 6040934.273,1])#1
hotel = np.array([4449525.439, 6041088.713,0]) 

#np.arctan2(y,x)

nordcorr3 = -np.arctan2(hotel[1]-theo3[1],hotel[0]-theo3[0])+np.pi*0.5
nordcorr4 = -np.arctan2(hotel[1]-theo4[1],hotel[0]-theo4[0])+np.pi*0.5

print(pyclamster.utils.rad2deg(nordcorr3))
print(pyclamster.utils.rad2deg(nordcorr4))

#az+nordcorrectur

if 0:
    plt.plot(0,0,'bo')
    plt.plot(theo4[0]-theo3[0],theo4[1]-theo3[1],'go')
    plt.plot(hotel[0]-theo3[0],hotel[1]-theo3[1],'rx')
    plt.xlim([-1400,0])
    plt.ylim([0,1400])
    plt.show()

#t3m1,t3m2,t4m1,t4m2
datafiles = ["./Theo_daten/rot/000R_20160901_110235.td4",
             "./Theo_daten/rot/000R_20160901_112343.td4",
             "./Theo_daten/gelb/000G_20160901_110235.td4",
             "./Theo_daten/gelb/000G_20160901_112344.td4"]


time1 = []
star1 = []
azim1 = []
elev1 = []

for k in range(len(datafiles)):
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
            azim1[-1].append(pyclamster.utils.deg2rad(float(line[7:13])))
            elev1[-1].append(pyclamster.utils.deg2rad(float(line[15:20])))
time = []
star = []
azim = []
elev = []

for i in range(len(datafiles)):
    time.append(np.array(time1[i]))
    star.append(np.array(star1[i]))
    azim.append(np.array(azim1[i]))
    elev.append(np.array(elev1[i]))
    #print('###')
    #print(np.sum(elev1[i]))
    #print(np.sum(azim1[i]))

maxlen = min([len(time[i]) for i in [0,2]])
m1 = np.empty([6,maxlen])
m1[0,:] = time[0][:maxlen]
m1[1,:] = star[0][:maxlen]+star[2][:maxlen]
m1[2,:] = pyclamster.utils.pos_rad(azim[0][:maxlen]+nordcorr3)
m1[3,:] = pyclamster.utils.pos_rad(azim[2][:maxlen]+nordcorr4)
m1[4,:] = elev[0][:maxlen]
m1[5,:] = elev[2][:maxlen]

#m1_theo3 = Coordinates3d(azimuth=pyclamster.deg2rad(m1[2,]),
#    elevation=pyclamster.deg2rad(m1[4,]))
#m1_theo4 = Coordinates3d(azimuth=pyclamster.deg2rad(m1[3,]),
#    elevation=pyclamster.deg2rad(m1[5,]))

maxlen = min([len(time[i]) for i in [1,3]])
m2 = np.empty([6,maxlen])
m2[0,:] = time[1][:maxlen]
m2[1,:] = star[1][:maxlen]+star[3][:maxlen]
m2[2,:] = pyclamster.utils.pos_rad(azim[1][:maxlen]+nordcorr3)
m2[3,:] = pyclamster.utils.pos_rad(azim[3][:maxlen]+nordcorr4)
m2[4,:] = elev[1][:maxlen]
m2[5,:] = elev[3][:maxlen]

#m2_theo3 = Coordinates3d(azimuth=pyclamster.deg2rad(m2[2,]),
#    elevation=pyclamster.deg2rad(m2[4,]))
#m2_theo4 = Coordinates3d(azimuth=pyclamster.deg2rad(m2[5,]),
#    elevation=pyclamster.deg2rad(m2[3,]))



def doppelanschnitt(azi1,azi2,ele1,ele2,pos1,pos2,col=['b']):
    e1 = np.array([np.sin(azi1)*np.cos(ele1),
                   np.cos(azi1)*np.cos(ele1),
                   np.sin(ele1)])

    e2 = np.array([np.sin(azi2)*np.cos(ele2),
                   np.cos(azi2)*np.cos(ele2),
                   np.sin(ele2)])

    n = np.cross(e1,e2)
    n = n/np.linalg.norm(n)
    #print(e1)
    #print(e2)
    #print(n)
    #print(pos1-pos2)
    abc = np.linalg.solve(np.array([e1,e2,n]),np.array([pos1-pos2]).T)
    #abc = np.linalg.lstsq(np.array([e1,e2,n]),np.array([pos1-pos2]).T)[0]
    #print(abc)
    position = np.array(pos1 - abc[0] * e1 - .5 * abc[2] * n)

    if 1: #test view direction and calced position
        #plt.figure()
        x = 0
        y = 2
        p1p = pos1-pos2
        p2p = pos2-pos2
        e1p = e1*50+p1p
        e2p = e2*50+p2p
        ppp = position-pos2
    
        #print(p1p)
        #print(p2p)
        #print(e1p)
        #print(e2p)
        #plt.plot(p1p[x],p1p[y],'go')
        #plt.plot(p2p[x],p2p[y],'ro')
        plt.plot([p1p[x],e1p[x]],[p1p[y],e1p[y]])
        plt.plot([p2p[x],e2p[x]],[p2p[y],e2p[y]])
        plt.plot(ppp[x],ppp[y],col[0]+'x')
        #plt.show()

    return position

pos1 = []
pos2 = []
for i in range(55):#range(55):
    if m1[1,i] == 0:
        pos1.append(np.array(doppelanschnitt(m1[2,i],m1[3,i],m1[4,i],m1[5,i],np.array(theo3),np.array(theo4))))
    if m2[1,i] == 0:
        pos2.append(np.array(doppelanschnitt(m2[2,i],m2[3,i],m2[4,i],m2[5,i],np.array(theo3),np.array(theo4),col=['g'])))

def theo_plot(xyz,pos1,pos2,zero_point=np.array([0,0,0]),col=['b','g']):
    if not isinstance(xyz,list):
        xyz = [xyz]
    for i,xi in enumerate(xyz):
        plt.plot(np.array(xi).T[0]-zero_point[0],np.array(xi).T[2]-zero_point[1],col[i]+'x')
    plt.plot(pos1[0]-zero_point[0],pos1[1]-zero_point[1],'ro')
    plt.plot(pos2[0]-zero_point[0],pos2[1]-zero_point[1],'go')

if 0: #test azi and elev
    plt.figure()
    plt.plot(m1[2,:])
    plt.plot(m1[3,:])
    plt.title('azi1')
    plt.figure()
    plt.plot(m2[2,:])
    plt.plot(m2[3,:])
    plt.title('azi2')
    plt.figure()
    plt.plot(m1[4,:])
    plt.plot(m1[5,:])
    plt.title('ele1')
    plt.figure()
    plt.plot(m2[4,:])
    plt.plot(m2[5,:])
    plt.title('ele2')
    plt.show()

#plt.figure()
#plt.title('m1')
#theo_plot([pos1,pos2],theo3,theo4,theo4)
#plt.figure()
#plt.title('m2')
#theo_plot(pos2,theo3,theo4,theo4)
plt.show()


