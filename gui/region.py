# -*- coding: utf-8 -*-
"""
Created on 09.06.16
Created for pyclamster-gui
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
import copy

# External modules
import numpy as np

# Internal modules

__version__ = "0.1"

class region(object):
    # image_margins = [left,right,top,bottom]
    def __init__(self,center=None,image_margins=None,image_data=None,region_data=None):
        self.center = np.array([0,0])
        self.data = None # [[x, y, r, g, b], ...] (x,y based on region center)

        if isinstance(center, np.ndarray):
            if center.size == 2:
                self.center = center
        elif isinstance(image_margins, np.ndarray):
            if image_margins.size == 4:
                self.center = self._calc_image_center(image_margins)

        if isinstance(region_data,np.ndarray):
            if region_data.shape[1] == 5 and len(region_data.shape) == 2:
                self.data = region_data
        elif isinstance(image_data,np.ndarray):
            if image_data.shape[1] == 5 and len(image_data.shape) == 2:
                self.data = image_data 
                self.data[:,0] = self.data[:,0] - self.center[0]
                self.data[:,1] = self.data[:,1] - self.center[1]
       

    def addData(self, image, x, y, center = None):
        if not isinstance(center,np.ndarray):
            center = self.center
        img_x = center[0] + x
        img_y = center[1] + y
        if not isinstance(self.data,np.ndarray):
            self.data = np.array([x,y,image[img_x,img_y,0],image[img_x,img_y,1],image[img_x,img_y,2]]).T
        else:
            # remove already added entries
            not_existing_entries = np.array([not [z[0],z[1]] in self.data[:,0:2].tolist() for z in zip(x,y)])
            x = x[not_existing_entries]
            y = y[not_existing_entries]
            # get new data that should be added
            new_data = np.array([x,y,image[img_x,img_y,0],image[img_x,img_y,1],image[img_x,img_y,2]]).T
            # append existing data
            self.data = np.append(self.data,new_data,0)
   
    def removeData(self, x, y, center_offset = None):
        if not isinstance(center_offset,np.ndarray):
            center_offset = np.array([0,0])
        x = x + center_offset[0]
        y = y + center_offset[1]
        # find existing entries
        z = np.array([x,y]).T
        if len(z.shape) == 1:
            z = np.array([z])
        existing_entries = np.array([z[i].tolist() in self.data[:,0:2].tolist() for i in range(len(z))])
        z = z[existing_entries]
        # define which entries to keep
        keep_entries = [not di in z.tolist() for di in self.data[:,0:2].tolist()]
        if any(keep_entries):
            self.data = self.data[np.array(keep_entries)]
        else:
            self.data = None
         

    def addRegion(self, region, center_offset = None):
        new_data = copy.copy(region.data)
        if not isinstance(center_offset,np.ndarray):
            center_offset = region.center - self.center
        new_data[:,0] = new_data[:,0] + center_offset[0]
        new_data[:,1] = new_data[:,1] + center_offset[1]
        # find not existing entries
        not_existing_entries = np.array([not zi in self.data[:,0:2].tolist() for zi in new_data[:,0:2].tolist()])
        self.data = np.append(self.data,new_data[not_existing_entries],0)


    def removeRegion(self, region, center_offset = None):
        rm_x = region.data[:,0]
        rm_y = region.data[:,1]
        if not isinstance(center_offset, np.ndarray):
            center_offset = region.center - self.center 
        self.removeData(rm_x, rm_y, center_offset)

    def cropImageRegion(self, image, center = None): #TODO
        if not isinstance(center,np.ndarray):
            center = self.center
        pass

    def exportToMask(self, image_margins, center = None): #TODO
        if not isinstance(center,np.ndarray):
            center = self.center
        pass

    def _calc_image_center(self, image_margins):
        x = (image_margins[1]-image_margins[0])*.5
        y = (image_margins[3]-image_margins[2])*.5
        return np.array([x,y],dtype=int)

if __name__ == '__main__':

    import numpy as np

    r  = region(region_data=np.array([[1,1,0,99,99]]),center=np.array([960,960]))
    r2 = region(region_data=np.array([[1,1,99,0,99]]),center=np.array([0,0]))
    r3 = region(image_data =np.array([[1,1,99,99,0]]),image_margins = np.array([0,1920,0,1920]))

    r.addRegion(r2)
    r.addRegion(r2,center_offset = np.array([10,100]))






