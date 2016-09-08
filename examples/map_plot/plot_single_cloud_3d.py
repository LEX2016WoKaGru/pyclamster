from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pyclamster

x = np.linspace(3640000,3643850,num=25)
y = np.linspace(6040000,6045410,num=25)
z = np.linspace(0,15000,num=25)
lons,lats = pyclamster.positioning.Projection().xy2lonlat(x,y)
pyclamster.positioning.plot_results3d(lons,lats,z)
plt.show()

