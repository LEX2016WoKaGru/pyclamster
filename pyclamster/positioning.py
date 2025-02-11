# -*- coding: utf-8 -*-
"""
Created on 03.09.16

Created for pyclamster

    Copyright (C) {2016}

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
# System modules
import logging
import copy


# External modules
import numpy as np
import scipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Internal modules
from . import coordinates
from . import utils

logger = logging.getLogger(__name__)

__version__ = "0.1"


def doppelanschnitt(azi1,azi2,ele1,ele2,pos1,pos2,plot_info=False):
    """
    calculate the 3d position via "doppelanschnitt"

    args:
        azi1, azi2, ele1, ele2 (float): SINGLE values of azimuth and elevation
            of different devices at positions 1 and 2 (radians)
        pos1, pos2 (np.array) [x,y,z] of different devices at positions
            1 and 2 (metres)

    returns:
        
    """

    e1 = np.array([np.sin(azi1)*np.cos(ele1),
                   np.cos(azi1)*np.cos(ele1),
                   np.sin(ele1)])

    e2 = np.array([np.sin(azi2)*np.cos(ele2),
                   np.cos(azi2)*np.cos(ele2),
                   np.sin(ele2)])

    n = np.cross(e1,e2,axis=0)
    n = n/np.linalg.norm(n)

    a_s = np.array([e1,-e2,n]).T
    b_s = (np.array(pos2)-np.array(pos1)).T
    a,b,c = np.linalg.solve(a_s, b_s)

    logger.debug("minimum distance: {} m".format(c))

    position = np.array(pos1 + a * e1 + n * 0.5 * c)
    
    var_list = [e1 ,e2, n, a, c]
    if plot_info:
        return position, var_list
    else:
        return position

# multiple values
def doppelanschnitt_Coordinates3d(aziele1,aziele2,pos1,pos2,plot_info=False):
    """
    calculate 3d position based on Coordinates3d

    args:
        aziele1,aziele2 (Coordinates3d): coordinates (azimuth/elevation) of 
            devices 1 and 2. These have to be northed
        pos1, pos2 (Coordinates3d): length 1 coordinates (x,y,z) of devices
            1 and 2

    returns:
        positions (Coordinates3d): (x,y,z) positions taken from
            Doppelanschnitt.
    """
    ae1 = copy.deepcopy(aziele1)
    ae2 = copy.deepcopy(aziele2)
    logger.debug("copied aziele1:\n{}".format(ae1))
    logger.debug("copied aziele2:\n{}".format(ae2))
    # turn to north
    ae1.fill(azimuth=ae1.azimuth, elevation=ae1.elevation, radius=1)
    ae2.fill(azimuth=ae2.azimuth, elevation=ae2.elevation, radius=1)

    # convert given positions to numpy array
    position1 = np.array([pos1.x,pos1.y,pos1.z])
    logger.debug("position1: \n{}".format(position1))
    position2 = np.array([pos2.x,pos2.y,pos2.z])
    logger.debug("position2: \n{}".format(position2))

    # loop over all azimuth/elevation values
    x = [];y = [];z = [];var_list = [] # start with empty lists
    for x1,y1,z1,x2,y2,z2 in zip(
        ae1.x.ravel(), ae1.y.ravel(), ae1.z.ravel(),
        ae2.x.ravel(), ae2.y.ravel(), ae2.z.ravel()):
        # calculate 3d doppelanschnitt position

        e1 = np.array([x1,y1,z1])
        e2 = np.array([x2,y2,z2])

        n = np.cross(e1,e2)
        n = n/np.linalg.norm(n)
        a_s = np.array([e1,-e2,n])
        b_s = (np.array(position2)-np.array(position1))
        try:
            a,b,c = scipy.linalg.lstsq(a_s.T, b_s)[0]
            #a,b,c = np.linalg.solve(a_s.T, b_s)
            xyz = np.array(position1 + a * e1 + n * 0.5 * c)
        except scipy.linalg.LinAlgError:
            a,b,c = np.nan, np.nan, np.nan
            xyz = [np.nan,np.nan,np.nan]
        except ValueError:
            a,b,c = np.nan, np.nan, np.nan
            xyz = [np.nan,np.nan,np.nan]
    
        logger.debug("minimum distance: {} m".format(c))

        var_list.append([e1 ,e2, n, a, c])

        x.append(xyz[0])
        y.append(xyz[1])
        z.append(xyz[2])

    # merge new coordinates
    out = coordinates.Coordinates3d(x=x,y=y,z=z,shape = ae1.shape)
    if plot_info:
        return out,np.array(var_list).T.tolist()
    else:
        return out

def doppelanschnitt_plot(title,position,var_list,pos1_in,pos2_in,col=['r','g','k','b'],
                         plot_view=False,plot_position=False,plot_n=False):
    """
    plots the output of doppelanschnitt.
    Args:
        position(Coordinates3d): result of 'doppelanschnitt'
        var_list(list): contains debug information ouput 
                        of 'doppelanschnitt'
        pos1_in, pos2_in(Coordinates3d): object containing position
    """


    x = 0
    y = 1
    z = 2

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title(title)
 
    pos1 = np.array([pos1_in.x,pos1_in.y,pos1_in.z])
    pos2 = np.array([pos2_in.x,pos2_in.y,pos2_in.z])
    e1 ,e2, n, a, c = var_list
    zero_point = pos1
    if isinstance(position,coordinates.Coordinates3d):
        try:
            zipped_pos = zip(position.x,position.y,position.z)
        except TypeError:
            zipped_pos = [[position.x,position.y,position.z]]
    else:
        try:
            zipped_pos = zip(position[0],position[1],position[2])
        except TypeError:
            zipped_pos = [[position[0],position[1],position[2]]]
    for i,pos_res in enumerate(zipped_pos):
        try:
            ppp  = np.array([pos_res[x],pos_res[y],pos_res[z]])
        except IndexError:
            ppp  = np.array(pos_res)
        zero_point = pos1
        
        ppp = ppp-zero_point
        p1p = pos1-zero_point
        p2p = pos2-zero_point
        try:
            e1p = e1[i]*a[i]+p1p
            e2p = e2[i]*a[i]+p2p
            n1p = n[i]*c[i]+ppp
            n2p = n[i]*c[i]+ppp
        except IndexError:
            e1p = e1*a+p1p
            e2p = e2*a+p2p
            n1p = n*c+ppp
            n2p = n*c+ppp
        
        if plot_view:
            ax.plot([p1p[x],e1p[x]],[p1p[y],e1p[y]],[p1p[z],e1p[z]],col[0], label='cam3')
            ax.plot([p2p[x],e2p[x]],[p2p[y],e2p[y]],[p2p[z],e2p[z]],col[1], label='cam4')
        if plot_n:
            ax.plot([n1p[x],n2p[x]],[n1p[y],n2p[y]],[n1p[z],n2p[z]],col[2])
        if plot_position:
            ax.plot([ppp[x]]       ,[ppp[y]]       ,[ppp[z]]       ,col[3]+'x')
    plt.legend()
    return ax
        

class Projection(object):
    def __init__(self, zone=32):
        try:
            import pyproj
        except:
            raise ImportError('pyproj isn\'t installed yet')

        proj = '+proj=tmerc +lat_0=0 +lon_0=9 +k=1 +x_0=3500000 +y_0=0 +ellps=bessel +datum=potsdam +units=m +no_defs'
        self.p = pyproj.Proj(proj)

    def lonlat2xy(self, lon, lat, return_coordinates=False):
        """
        Method to calculate x, y coordinates from latitudes and longitudes.
        Args:
            lon (float/numpy array): Longitudes in decimal degree as array or as float.
            lat (float/numpy array): Latitudes in decimal degree as array or as float.
            return_coordinates (optional[bool]): If the output coordinates should be a Coordinates3d instance. Default is False.
        Returns:
            pos (tuple/Coordinates3d): The x, y position as tuple or as Coordinates3d instance, depending on the coordinates argument.
        """
        pos = self.p(lon, lat)
        if return_coordinates:
            pos = coordinates.Coordinates3d(
                x=pos[0],
                y=pos[1],
                azimuth_offset=3 / 2 * np.pi,
                azimuth_clockwise=True
            )
        return pos

    def xy2lonlat(self, x, y=None):
        """
        Method to calculate longitude and latitude coordinates from x and y coordinates.
        Args:
            x (Coordinates3d/float/numpy array): The x coordinate as numpy array or as float. If this is a Coordinates3d instance, the x and y coordinates are calculated out of this instance.
            y (optional[float/numpy array): The x coordinate as numpy array or as float. This argument is unnecessary if the x coordiante is a Coordinates3d instance. Default is None.
        Returns:
            pos (tuple[float/numpy array]): The longitudes and latitudes as tuple of floats or as tuple of numpy arrays if the input arguments are also a numpy array.
        """
        if isinstance(x, coordinates.Coordinates3d):
            coords = x
            x = coords.x
            y = coords.y
        pos = self.p(x, y, inverse=True)
        return pos

def plot_results3d(lons,lats,z):
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from mpl_toolkits.mplot3d import Axes3D

    llcrnrlon = 11.
    urcrnrlon = 11.3
    llcrnrlat = 54.4
    urcrnrlat = 54.55
    if np.min(lons)> llcrnrlon: llcrnrlon = np.min(lons)
    if np.max(lons)> urcrnrlon: urcrnrlon = np.max(lons)
    if np.min(lats)> llcrnrlat: llcrnrlat = np.min(lats)
    if np.max(lats)> urcrnrlat: urcrnrlat = np.max(lats)
  

    # create new figure, axes instances.
    fig=plt.figure()
    ax=Axes3D(fig)
    ax.set_title('stereo cam results')
    
    # setup mercator map projection.
    m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,
                resolution='i',projection='merc')
    x, y = m(lons,lats)
    #m.fillcontinents(zorder=1)
    ax.add_collection3d(m.drawcoastlines())
    # draw parallels
    #m.drawparallels(np.arange(54,55,0.1),labels=[1,1,0,1])
    # draw meridians
    #m.drawmeridians(np.arange(10,12,0.1),labels=[1,1,0,1])
    for xi,yi,zi in zip(x,y,z):
        ax.plot([xi],[yi],[zi],'o',color=utils.calc_color(zi,0,15000))
        ax.plot([xi,xi],[yi,yi],[0,zi],color=utils.calc_color(zi,0,15000))
        ax.plot([xi],[yi],[0],'x',color=utils.calc_color(zi,0,15000))
    # Now adding the colorbar
    ax.set_zlabel('Hoehe [m]')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    return ax,m

def plot_results2d(lons,lats,z):
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from mpl_toolkits.mplot3d import Axes3D

    llcrnrlon = 11.
    urcrnrlon = 11.3
    llcrnrlat = 54.4
    urcrnrlat = 54.55
    if np.min(lons)> llcrnrlon: llcrnrlon = np.min(lons)
    if np.max(lons)> urcrnrlon: urcrnrlon = np.max(lons)
    if np.min(lats)> llcrnrlat: llcrnrlat = np.min(lats)
    if np.max(lats)> urcrnrlat: urcrnrlat = np.max(lats)

    # create new figure, axes instances.
    fig=plt.figure()
    #ax=Axes3D(fig)
    ax = fig.add_subplot(111)
    ax.set_title('stereo cam results')
    
    # setup mercator map projection.
    m = Basemap(llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,
                resolution='i',projection='merc')
    x, y = m(lons,lats)
    m.fillcontinents(zorder=1)
    m.drawcoastlines()
    # draw parallels
    m.drawparallels(np.arange(54,55,0.1),labels=[1,1,0,1])
    # draw meridians
    m.drawmeridians(np.arange(10,12,0.1),labels=[1,1,0,1])
    #print(x,y,z)
    sc = ax.scatter(x,y,c=z,cmap='RdBu',vmin=0,vmax=15000,zorder=10)
    # Now adding the colorbar
    cb = plt.colorbar(mappable=sc,cmap='RdBu',ax=ax,pad=0.13)
    cb.set_clim(0,15000)
    cb.set_label('Hoehe [m]')
    return ax,m

