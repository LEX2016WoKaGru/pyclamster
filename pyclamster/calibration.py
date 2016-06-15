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
    """
    class to hold calibration parameters
    """
    def __init__(self, 
                 center_row  = None,
                 center_col  = None, 
                 north_angle = None,
                 f = None, 
                 alpha = None
                 ):
        """
        class constructor
        args:
            center_row,center_col (numeric): center position of optical axis
            north_angle (numeric): azimuth offset (see Coordinate3d docs)
            f (numeric): proportionality factor for projected image radius
            alpha (numeric): empirical exponent for projected image radius
        """
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
        try:    estimate = estimate.parameters
        except: pass
        est_row_c, est_col_c, est_north, est_f, est_alpha = estimate
        row, col = self.pixel_coords.y, self.pixel_coords.x
        azi, ele = self.sun_coords.azimuth, self.sun_coords.elevation
        #logger.debug("row: {row}, col: {col}, azi: {azi}, ele: {ele}".format(
            #row=row,col=col,azi=azi,ele=ele))
        #logger.debug("row_c: {row_c}, col_c: {col_c}, f: {f}, alpha: {a}".format(
            #row_c=est_row_c,col_c=est_col_c,f=est_f,a=est_alpha))
    
        # set estimate of azimuth offset
        self.pixel_coords.azimuth_offset = est_north

        ### calculate elevation residual ###
        #logger.debug("estimated fisheye image radius: {}".format(est_radius_fisheye))
        # radius from estimated center for each measured input dataset
        est_radius_image = np.sqrt((row-est_row_c)**2+(col-est_col_c)**2) ** est_alpha
        # estimated elevation according to radial projection
        est_ele_image = est_radius_image / est_f
        #logger.debug("estimated image radius: {}".format(est_radius_image))
        # elevation residual
        res_ele = ele - est_ele_image

        ### calculate azimuth residual ###
        res_azi = azi - self.pixel_coords.azimuth
        
        # return resulting error
        return(abs(res_ele.mean())+abs(res_azi.mean()))



# calibrator class
class CameraCalibrator(object):
    def __init__(self, image, method=None):
        """
        class constructor
        args:
            image (pyclamster.Image): objective image
            method (str or None): optimization method. see 
                scipy.optimize.minimize documentation for further information.
                Defaults to None which is not recommended because the standard
                method varies between some scipy versions.
        """
        self.image  = image
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
            bounds=[(0,self.image.data.shape[0]), # row bound
                    (0,self.image.data.shape[1]), # col bound
                    (-4*np.pi,4*np.pi),           # north_angle bound
                    (0,np.Inf),                   # f bound
                    (0,1)]                        # alpha bound
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
