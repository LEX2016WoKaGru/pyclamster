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

# External modules

# Internal modules
#TODO import region

__version__ = "0.1"

class RegionSelectionFrame(object):
    def __init__(self,base_image=None,camera=None,path=None):        

        self.base_image = None
        self.camera = None

        if isinstance(path,str):
            #TODO check if path is file
            self.loadImage(path)
            #TODO check if path is a directory
            #     create pyclamster camera-object
        elif isinstance(base_image,pyclamster.image):
            self.base_image = base_image
        elif isinstance(camera,pyclamster.camera):
            self.camera = camera
     

    def loadImage(self,path):
        """
        load an image that should be worked on
        """
        pass

    def loadMask(self,path):
        """
        load a masks that should be worked with
        """
        pass

    def createRegionFromMask(self,mask):
        """
        create instance of class region by instance of class 'mask'
        """
        pass

    def markRegionMode(self):
        """
        switch mode to mark an image region to work with
        """
        pass
    
    def selectRegionMode(self):
        """
        switch mode to select a region to work with
        also possible to select in region-lable list 
        """ #TODO toggle to label list
        pass
   
    def selectRegion(self):
        """
        select a region by marking it on the displayed image
        """
        pass

    def appendRegion(self,region):
        """
        append the currently selected region by further marking
        """
        pass

    def connectRegions(self,regions):
        """
        connect the currently selected regions to be one region
        """
        pass
    
    def deleteRegions(self,regions):
        """
        delete currently selected regions
        """
        pass

    def cropRegions(self,regions):
        """
        create a 'maskedImage' showing only the marked region
        for every selected region
        """
        pass

    def exportRegions(self,regions):
        """
        export selected regions with lables as instance of class 'mask'
        """
        pass

    def regionStatLabelToggle(self):
        """
        toggles between showing region statistics and region lables
        """
        pass

    def overlayRegion(self,current_image,region):
        """
        add new region to currently shown image
        """
        pass

    def createRegionOverlay(self,image,regions):
        """
        add all selected regions to the base image
        """
        pass

