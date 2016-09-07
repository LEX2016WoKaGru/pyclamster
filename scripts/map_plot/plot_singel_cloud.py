from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pyclamster

x = 3641343.2618716503
y = 6042905.146374716
#z = 1999.9201277320833
z = 10000

x = np.linspace(3641300,3641350,num=25)
y = np.linspace(6042900,6042910,num=25)
z = np.linspace(0,15000,num=25)
lons,lats = pyclamster.positioning.Projection().xy2lonlat(x,y)

# create new figure, axes instances.
fig=plt.figure()
#ax=Axes3D(fig)
ax = fig.add_subplot(111)

# setup mercator map projection.
m = Basemap(llcrnrlon=11.,llcrnrlat=54.4,urcrnrlon=11.3,urcrnrlat=54.55,
            resolution='i',projection='merc')
x, y = m(lons,lats)
m.fillcontinents(zorder=1)
m.drawcoastlines()
# draw parallels
m.drawparallels(np.arange(54,55,0.1),labels=[1,1,0,1])
# draw meridians
m.drawmeridians(np.arange(10,12,0.1),labels=[1,1,0,1])
ax.set_title('Great Circle from New York to London')
print(x,y,z)
ax.scatter(x,y,c=z,cmap='winter',vmin=0,vmax=15000,zorder=10)
plt.colorbar()
plt.show()

