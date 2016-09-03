import matplotlib.pyplot as plt
import sys
import numpy as np
import pyclamster
from mpl_toolkits.mplot3d import Axes3D
from pyclamster.coordinates import Coordinates3d

print('import done...')

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

def doppelanschnitt(azi1,azi2,ele1,ele2,pos1,pos2,col=['r','g','k','b'],zero_point = [0,0,0],
                    ax=None,plot_view=False,plot_n=False,plot_position=False):

    e1 = np.array([np.sin(azi1)*np.cos(ele1),
                   np.cos(azi1)*np.cos(ele1),
                   np.sin(ele1)])

    e2 = np.array([np.sin(azi2)*np.cos(ele2),
                   np.cos(azi2)*np.cos(ele2),
                   np.sin(ele2)])

    n = np.cross(e1,e2,axis=0)
    n = n/np.linalg.norm(n)

    abc = np.linalg.solve(np.array([e1,e2,n]).T,np.array([pos1-pos2]).T)
    #abc2 = np.linalg.lstsq(np.array([e1,e2,n]),np.array([pos1-pos2]).T)[0]

    position = np.array(pos1 - abc[0] * e1 - n * 0.5 * abc[2])
    print(abc[2])
    if not ax is None: #test view direction and calced position
        x = 0
        y = 1
        z = 2

        ppp = position-zero_point
        p1p = pos1-zero_point
        p2p = pos2-zero_point
        e1p = e1*300+p1p
        e2p = e2*300+p2p
        n1p = n*10+ppp
        n2p = -n*10+ppp

        if plot_view:
            ax.plot([p1p[x],e1p[x]],[p1p[y],e1p[y]],[p1p[z],e1p[z]],col[0])
            ax.plot([p2p[x],e2p[x]],[p2p[y],e2p[y]],[p2p[z],e2p[z]],col[1])
        if plot_n:
            ax.plot([n1p[x],n2p[x]],[n1p[y],n2p[y]],[n1p[z],n2p[z]],col[2])
        if plot_position:
            ax.plot([ppp[x]]       ,[ppp[y]]       ,[ppp[z]]       ,col[3]+'x')

    return position

def plot_cam_pos(ax,pos,col=['b'],zero_point = [0,0,0]):
    ax.plot([pos[0]-zero_point[0]],[pos[1]-zero_point[1]],[pos[2]-zero_point[2]],col[0]+'o')


def plot_azi_ele(m):
    plt.figure()
    plt.plot(m[2,:])
    plt.plot(m[3,:])
    plt.title('azi')
    plt.figure()
    plt.plot(m[4,:])
    plt.plot(m[5,:])
    plt.title('ele')

######################################################

theo3 = np.array([4450909.840, 6040800.456,6])
theo4 = np.array([4450713.646, 6040934.273,1])
hotel = np.array([4449525.439, 6041088.713,0]) 

#t3m1,t4m1,t3m2,t4m2
datafiles = [["./Theo_daten/rot/000R_20160901_110235.td4",
              "./Theo_daten/gelb/000G_20160901_110235.td4"],
             ["./Theo_daten/rot/000R_20160901_112343.td4",
              "./Theo_daten/gelb/000G_20160901_112344.td4"]]

nordcorr3 = -np.arctan2(hotel[1]-theo3[1],hotel[0]-theo3[0])+np.pi*0.5
nordcorr4 = -np.arctan2(hotel[1]-theo4[1],hotel[0]-theo4[0])+np.pi*0.5


fig = plt.figure()
ax = Axes3D(fig)
plt.title('Doppelanschnitt (r=cam3; g=cam4)')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
position1 = []
position2 = []
m = read_theo_data(datafiles)
for i in [0,1]:
    for n in range(len(m[i][2,:])):
        m[i][2,n] = pyclamster.utils.pos_rad(m[i][2,n]+nordcorr3)
        m[i][3,n] = pyclamster.utils.pos_rad(m[i][3,n]+nordcorr4)

plot_cam_pos(ax,theo3,['r'],theo3)
plot_cam_pos(ax,theo4,['g'],theo3)
#plot_cam_pos(ax,hotel,['k'],theo3)
col = []
for mi in [0]:
    for i in range(1):
        if m[mi][1,i] == 0:
            position1.append(np.array(doppelanschnitt(m[mi][2,i],m[mi][3,i],m[mi][4,i],m[mi][5,i],
                                          np.array(theo3),np.array(theo4),zero_point = theo3,
                                          ax=ax,plot_view=True,plot_n=True,plot_position=True)))
plt.show()


