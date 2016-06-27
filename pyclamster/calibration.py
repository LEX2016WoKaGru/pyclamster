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
import copy

# External modules
import numpy as np
import scipy.optimize as optimize

# Internal modules


__version__ = "0.1"

# create logger
logger = logging.getLogger(__name__)


class CameraCalibrationParameters(object):
    """
    class to hold calibration parameters
    """
    def __init__(self, 
                 center_row  = None,
                 center_col  = None, 
                 north_angle = None,
                 r2 = None,
                 r4 = None,
                 r6 = None
                 ):
        """
        class constructor
        args:
            center_row,center_col (numeric): center position of optical axis
            north_angle (numeric): azimuth offset (see Coordinate3d docs)
            f (numeric): proportionality factor for projected image radius
        """
        self.parameters = (center_row, center_col, north_angle,r2,r4,r6)

    @property
    def parameters(self):
        parms = (self.center_row, self.center_col, 
                 self.north_angle, self.r2,self.r4,self.r6)
        return(parms)

    @parameters.setter
    def parameters(self, newparams):
        self.center_row, self.center_col, \
        self.north_angle,self.r2,self.r4,self.r6 = newparams

    # summary when converted to string
    def __str__(self):
        parameters = ("center_row","center_col",
                      "north_angle","r2","r4","r6")
        formatstring = ["=================================",
                        "| camera calibration parameters |",
                        "================================="]
        for param in parameters:
            formatstring.append("{:>11}: {}".format(param,getattr(self, param)))
        return("\n".join(formatstring))

class CameraCalibrationRadialFunction(object):
    def __init__(self, parameters):
        self.parameters = parameters

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        new = CameraCalibrationParameters()
        try:    new.parameters = value.parameters
        except: new.parameters = value
        self._parameters = new

    def __call__(self, elevation):
        e = elevation
        p = self.parameters
        #return p.r6*e**3+p.r4*e**2+p.r2*e**1
        return p.r6*np.tan(e/2)

class CameraCalibrationLossFunction(object):
    def __init__(self, pixel_coords, sun_coords, radial):
        """
        class constructor.
        args:
            pixel_coords (CarthesianCoordinates3d): carthesian coordinates
                (y=row,x=col) of sun pixel positions on image.
            sun_coords (SphericalCoordinates3D): spherical coordinates
                (azimuth, elevation) of sun positions.
            radial (callable): radial distortion funciton. Given the elevation 
                as parameter, return the (unified) image radius in pixel units.
                This does not need to be the exact radius, it may be
                (in all directions equally) scaled by an arbitrary factor.
        returns:
            calibration(CameraCalibrationLossFunction)
        """
        # copy over attributes
        self.pixel_coords = pixel_coords
        self.sun_coords   = sun_coords
        self.radial       = radial

    def __call__(self, estimate):
        """
        calculate loss function value (residuals) for a given estimate
        args:
            estimate (5-tuple or CameraCalibrationParameters): estimate
        returns:
            residual = numeric value
        """
        logger.debug(estimate)
        try:    estimate = estimate.parameters
        except: pass
        est_row_c, est_col_c, est_north,*radialp = estimate

        # local copy of coordinates
        pixel_coords = copy.deepcopy(self.pixel_coords) # a copy
        sun_coords   = copy.deepcopy(self.sun_coords) # a copy

        logger.debug("================== begin iteration =====================")
        # set the estimated center of the measured pixel coordinates
        pixel_coords.fill( 
            x=pixel_coords.x-est_col_c,
            y=pixel_coords.y-est_row_c)
        # turn the image system to the north angle
        pixel_coords.change_parameters(azimuth_offset=est_north,
            keep={'azimuth','radiush'})

            
        # calculate real (sun) x and y with the given center
        self.radial.parameters = estimate
        sun_coords.radiush = self.radial(sun_coords.elevation) 

        logger.debug("pixel_coords\n{}".format(pixel_coords))
        logger.debug("sun_coords\n{}".format(sun_coords))

        i = 1
        if i == 1: # optimize x/y matching (WORKS BEST)
            res = np.sqrt((sun_coords.x-pixel_coords.x) ** 2 + \
                      (sun_coords.y-pixel_coords.y) ** 2).mean()
        elif i == 2: # optimize radiush matching
            res = np.sqrt((sun_coords.radiush-pixel_coords.radiush) ** 2).mean()
        elif i == 3: # optimize x/y and radiush matching
            res = np.sqrt((sun_coords.x-pixel_coords.x) ** 2 + \
                          (sun_coords.y-pixel_coords.y) ** 2 + \
                          (sun_coords.radiush-pixel_coords.radiush) ** 2 ).mean()
            
        if res is np.ma.masked:
            raise Exception("Residual is masked. Something is wrong.")

        logger.debug("residual: {res}".format(res=res))
        logger.debug("================== end iteration =====================")

        return(res)



# calibrator class
class CameraCalibrator(object):
    def __init__(self,method=None):
        """
        class constructor
        args:
            method (str or None): optimization method. see 
                scipy.optimize.minimize documentation for further information.
                Defaults to None which is not recommended because the standard
                method varies between some scipy versions.
        """
        self.method = method

    def estimate(self, lossfunc, first_guess):
        """
        fit calibration parameters with respect to a lossfunction given a 
        first guess of parameters.
        args:
            lossfunc (CameraCalibrationLossFunction): the lossfunction
            first_guess (CameraCalibrationParameters): the initial guess
        """
        # empty parameterrs
        opt_estimate = CameraCalibrationParameters()
        # optimize parameters
        estimate = optimize.minimize(
            lossfunc,
            first_guess.parameters,
            method=self.method,
            bounds=[(0,1920), # row bound
                    (0,1920), # col bound
                    (0,2*np.pi),                  # north_angle bound
                    (-np.Inf,np.Inf),
                    (-np.Inf,np.Inf),
                    (-np.Inf,np.Inf)
                    ]                  
            )
         
        opt_estimate.parameters = estimate.x

        calibration = CameraCalibration(parameters = opt_estimate, 
            fit = estimate )

        return(calibration)


# calibration class
class CameraCalibration(object):
    def __init__(self, parameters, func=None, fit=None):

        # copy over attributes
        self.parameters  = parameters
        self.func        = func
        self.fit         = fit

    # return elevation
    def elevation(self):
        pass

    # return azimuth
    def azimuth(self):
        pass
