#!/usr/bin/env python3
import pyclamster
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('tfinn-poster')
#plt.ticklabel_format(style='sci', axis='x', scilimits=(-100000,100000))

rng = np.random.RandomState(42)

azi1, ele1 = 3.526, 0.636
azi2, ele2 = 3.567, 0.666

stdev = 0.1
points = 1000

azi1 = azi1+pyclamster.deg2rad(rng.normal(0,stdev,size=points))
ele1 = ele1+pyclamster.deg2rad(rng.normal(0,stdev,size=points))
azi2 = azi2+pyclamster.deg2rad(rng.normal(0,stdev,size=points))
ele2 = ele2+pyclamster.deg2rad(rng.normal(0,stdev,size=points))
#azi1 = pyclamster.deg2rad(azi1+rng.normal(0,stdev,size=points))
#ele1 = pyclamster.deg2rad(ele1+rng.normal(0,stdev,size=points))
#azi2 = pyclamster.deg2rad(azi2+rng.normal(0,stdev,size=points))
#ele2 = pyclamster.deg2rad(ele2+rng.normal(0,stdev,size=points))

theo1 = pyclamster.Coordinates3d(
    azimuth = azi1,
    elevation = ele1,
    azimuth_clockwise = True,
    azimuth_offset = 3/2*np.pi,
    elevation_type = "zenith"
    )

x, y = pyclamster.Projection().lonlat2xy(11.240817, 54.4947)
pos1 = pyclamster.Coordinates3d(
    x = x, y = y, z = 9
    )


theo2 = pyclamster.Coordinates3d(
    azimuth = azi2,
    elevation = ele2,
    azimuth_clockwise = True,
    azimuth_offset = 3/2*np.pi,
    elevation_type = "zenith"
    )

x, y = pyclamster.Projection().lonlat2xy(11.2376833, 54.495866)
pos2 = pyclamster.Coordinates3d(
    x = x, y = y, z = 0
    )
    
doppel,var_list_c3d = pyclamster.positioning.doppelanschnitt_Coordinates3d(
    aziele1 = theo1,
    aziele2 = theo2,
    pos1    = pos1,
    pos2    = pos2,
    plot_info=True
    )
#ax = pyclamster.doppelanschnitt_plot('c3d single point by hand',doppel,var_list_c3d,pos1,pos2,plot_view=1,plot_position=1)

ax = doppel.plot3d()
ax.scatter3D(pos1.x,pos1.y,pos1.z,color='red', label='cam3')
ax.scatter3D(pos2.x,pos2.y,pos2.z,color='green', label='cam4')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.legend()
#ax.set_xlim([-100, 300])
#ax.set_ylim([-100, 300])
#ax.set_zlim([-100, 300])
print(np.min(doppel.x), np.median(doppel.x), np.max(doppel.x))
print(np.min(doppel.y), np.median(doppel.y), np.max(doppel.y))
print(np.min(doppel.z), np.median(doppel.z), np.max(doppel.z))
binwidth = 10
fig, ax = plt.subplots()
ax.hist(doppel.z, range=(0, 10000), bins=100, label='Height distribution')
ax.axvline(x=np.median(doppel.z), color='darkred', label='Median')
ax.axvline(x=np.mean(doppel.z), color='green', label='Mean')
#ax.set_xlim([min(0, np.min(doppel.z)), max(500, np.max(doppel.z))])
ax.set_ylabel('Occurrence / {0:d} draws'.format(points))
ax.set_xlabel('Height [m], {0:d} m per bin'.format(binwidth))
plt.legend()
plt.show()