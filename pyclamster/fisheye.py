# -*- coding: utf-8 -*-
"""
Created on 30.05.16

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

# External modules
import numpy as np

# Internal modules


__version__ = "0.1"

# create logger
logger = logging.getLogger(__name__)


# fisheye class
class FisheyeProjection(object):
    """
    class for utils to handle fisheye projections
    """
    def __init__(self, projection=None):
        # currently implemented fisheye projections
        self._implemented_projections = ("equidistant")

        self.projection = projection


    ##################
    ### Properties ###
    ##################
    @property
    def projection(self):   
        return self._projection

    @projection.setter
    def projection(self, newprojection):
        if not newprojection is None:
            if newprojection in self._implemented_projections:
                self._projection = newprojection
            else: # if not implemented, raise error
                raise ValueError("unimplemented fisheye projection '{}'".format(
                    newprojection))
        else:
            self._projection = None

    ###############
    ### Methods ###
    ###############
    def createElevation(self, shape, center=None, maxangle=None, projection=None):
        """
        create a np.ndarray that holds an elevation for each pixel of an image
        according to fisheye projection type

        args:
            shape(2-tuple of int): shape of image (width, height)
            center(optional[2-tuple of int]): center/optical axis position on image (row,col).
                if not specified, the center of the image is assumed
            maxangle(optional[float]): the maximum angle (rad) from optical axis/center visible on the image

        returns:
            np.ndarray of shape (width, height) with elevation values
        """
        # shape
        try:
            width, height = shape
        except:
            raise ValueError("shape argument not defined as 2-tuple.")

        # center
        try:
            center_row, center_col = center
        except:
            logger.debug("center not specified as 2-tuple, assuming image center")
            center_row, center_col = int(height/2),int(width/2) # image center

        # maxangle
        # maxangle = maxangle :-)
            
        # projection
        if projection is None: # projection defined?
            projection = self.projection # default projection from object
        if projection is None: # if projection is still not defined
            raise ValueError("no fisheye projection specified.")

            
        # now create arrays based on projection
        if projection == "equidistant":
            # equidistant
            pass
        else:
            logger.debug(" ".join(("unimplemented fisheye projection '{}'",
                                   "You should never see this...")
                                   ).format(projection))


        

###############
### Example ###
###############
if __name__ == '__main__':
    pass
