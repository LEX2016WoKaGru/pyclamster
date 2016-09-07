from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pyclamster

x = np.linspace(3640000,3643850,num=25)
y = np.linspace(6040000,6045410,num=25)
z = np.linspace(0,15000,num=25)
lons,lats = pyclamster.positioning.Projection().xy2lonlat(x,y)

def calc_color(val,vmin,vmax):
    if val < vmin:
        val = vmin
    if val > vmax: 
        val = vmax
    c = (val-vmin)/(vmax-vmin)
    r,g,b = np.array([1-c,0,c])*99
    r = str(int(r))
    g = str(int(g))
    b = str(int(b))
    if len(r) == 1: r = '0'+r;
    if len(g) == 1: g = '0'+g;
    if len(b) == 1: b = '0'+b;
    return '#'+r+g+b

def plot_results(x,y,z):
    
    lons,lats = pyclamster.positioning.Projection().xy2lonlat(x,y)
    
    # create new figure, axes instances.
    fig=plt.figure()
    ax=Axes3D(fig)
    ax.set_title('stereo cam results')
    
    # setup mercator map projection.
    m = Basemap(llcrnrlon=11.,llcrnrlat=54.4,urcrnrlon=11.3,urcrnrlat=54.55,
                resolution='i',projection='merc')
    x, y = m(lons,lats)
    #m.fillcontinents(zorder=1)
    ax.add_collection3d(m.drawcoastlines())
    # draw parallels
    #m.drawparallels(np.arange(54,55,0.1),labels=[1,1,0,1])
    # draw meridians
    #m.drawmeridians(np.arange(10,12,0.1),labels=[1,1,0,1])
    for xi,yi,zi in zip(x,y,z):
        ax.plot([xi],[yi],[zi],'o',color=calc_color(zi,0,15000))
        ax.plot([xi,xi],[yi,yi],[0,zi],color=calc_color(zi,0,15000))
        ax.plot([xi],[yi],[0],'x',color=calc_color(zi,0,15000))
    # Now adding the colorbar
    ax.set_zlabel('Hoehe [m]')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

plt.show()

