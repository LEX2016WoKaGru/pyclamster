# -*- coding: utf-8 -*-
"""
Created on 14.06.16

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
import scipy.optimize as optimize

# Internal modules


__version__ = "0.1"

# create logger
logger = logging.getLogger(__name__)


class CameraCalibrationParameters(object):
    def __init__(self, 
                 center_row  = None,
                 center_col  = None, 
                 north_angle = None,
                 f = None, 
                 alpha = None
                 ):

        self.parameters = (center_row, center_col, north_angle, f, alpha)

    @property
    def parameters(self):
        parms = (self.center_row, self.center_col, 
                 self.north_angle, self.f, self.alpha)
        return(parms)

    @parameters.setter
    def parameters(self, newparams):
        self.center_row, self.center_col, \
        self.north_angle, self.f, self.alpha = newparams

    # summary when converted to string
    def __str__(self):
        parameters = ("center_row","center_col",
                      "north_angle","f","alpha")
        formatstring = ["=================================",
                        "| camera calibration parameters |",
                        "================================="]
        for param in parameters:
            formatstring.append("{:>11}: {}".format(param,getattr(self, param)))
        return("\n".join(formatstring))


class CameraCalibrationFunction(object):
    def __init__(self, azimuth_func, elevation_func):
        self.azimuth_func   = azimuth_func 
        self.elevation_func = elevation_func

    def azimuth(self, parameters):
        try: parameters = parameters.parameters
        except: pass
        self.azimuth_func(parameters)

    def elevation(self, parameters):
        try: parameters = parameters.parameters
        except: pass
        self.elevation_func(parameters)

    def __call__(self, parameters):
        try: parameters = parameters.parameters
        except: pass
        # calculate and return
        return np.array([self.azimuth(parameters),
                         self.elevation(parameters)])

class CameraCalibrationLossFunction(object):
    def __init__(self, known_row, known_col, known_azimuth, known_elevation,
                 radial):
        # copy over attributes
        self.row    = known_row
        self.col    = known_col
        self.azi    = known_azimuth
        self.ele    = known_elevation
        self.radial = radial

    def __call__(self, estimate):
        try:    estimate = estimate.parameters
        except: pass
        est_row_c, est_col_c, est_north, est_f, est_alpha = estimate
    
        ### calculate elevation residual ###
        # radius from estimated center for each measured input dataset
        r_est = np.sqrt((self.row-est_row_c)**2+(self.col-est_col_c)**2) / est_f
        # "real" radius when taking elevation directly
        r_real = self.radial(self.ele) ** est_alpha
        # elevation residual
        res_ele = r_est - r_real

        ### calculate azimuth residual ###
        # TODO: calculate azimuth residual
        
        # return resulting error
        return(res_ele.mean())



# calibrator class
class CameraCalibrator(object):
    def __init__(self, image):
        self.image = image

    def estimate(self, lossfunc, first_guess):
        # empty parameterrs
        opt_estimate = CameraCalibrationParameters()
        # optimize parameters
        estimate = optimize.minimize(
            lossfunc,
            first_guess.parameters,
            bounds=[(0,self.image.data.shape[0]), # row bound
                    (0,self.image.data.shape[1]), # col bound
                    (-4*np.pi,4*np.pi),           # north_angle bound
                    (0,np.Inf),                   # f bound
                    (0,1)]                   # alpha bound
            )
         
        opt_estimate.parameters = estimate.x
        return(opt_estimate)


# calibration class
class CameraCalibration(object):
    def __init__(self, parameters, func):

        # copy over attributes
        self.parameters  = parameters
        self.func        = func

    # return elevation
    def elevation(self):
        pass

    # return azimuth
    def azimuth(self):
        pass
