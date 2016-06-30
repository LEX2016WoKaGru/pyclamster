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
import scipy.interpolate as interpolate

# Internal modules
from . import coordinates


__version__ = "0.1"

# create logger
logger = logging.getLogger(__name__)


class CameraCalibrationParameters(object):
    """
    class to hold calibration parameters
    You may alter the bounds attribute, which is a list of (upper,lower) bounds
    of the parameters. They are initially set to sensible values, i.e.
    (0,Inf) for the center positions and (0,2*pi) for the north_angle.
    The bounds may improve the optimization for some optimization methods.
    """
    def __init__(self, center_row=0, center_col=0, north_angle=0,
                 *radialp):
        """
        class constructor
        args:
            center_row,center_col (numeric): center position of optical axis
            north_angle (numeric): azimuth offset (see Coordinates3d docs)
            radialp (numeric): arbitrary number of radial distortion parameters 
                that have to be sufficient for the radial distortion function.
                They will be named r0 ... rn.
        """
        # set the parameters
        parameters = [center_row, center_col, north_angle]
        for p in radialp: parameters.append(p)
        self.parameters = parameters

    # make it iterable
    def __iter__(self):
        try: del self.current # reset counter
        except: pass
        return self

    def __next__(self):
        try:    self.current += 1 # try to count up
        except: self.current  = 0 # if that didn't work, start with 0
        if self.current >= len(self.parameters):
            del self.current # reset counter
            raise StopIteration # stop the iteration
        else:
            return self.parameters[self.current]
        
    # make it indexable
    def __getitem__(self,key):
        return self.parameters[key]

    @property
    def parameter_names(self):
        try:    return self._parameter_names
        except: return []

    @parameter_names.setter
    def parameter_names(self, newnames):
        # If new parameter names
        if not newnames == self.parameter_names:
            # set new parameter names
            self._parameter_names = newnames
            # set all parameters to None
            self.bounds = []
            for p in self.parameter_names:
                setattr(self,p,0) # set parameter value to 0
                # set bounds to infinity
                self.bounds.append((-np.Inf,np.Inf))
            # initially set sensible bounds
            self.bounds[0] = (0,np.Inf)
            self.bounds[1] = (0,np.Inf)
            self.bounds[2] = (0,2*np.pi)


    @property
    def parameters(self):
        parms = []
        for p in self.parameter_names:
            parms.append(getattr(self,p))
        return(parms)

    @parameters.setter
    def parameters(self, newparams):
        # create a list of parameter names
        parameter_names = ["center_row","center_col","north_angle"]
        for i,p in enumerate(newparams[3:]):
            parameter_names.append("r{}".format(i))

        # set the parameter names
        self.parameter_names = parameter_names

        # set the new parameters
        for name,val in zip(self.parameter_names,newparams):
            setattr(self,name,val)

    # summary when converted to string
    def __str__(self):
        formatstring = ["===========================================",
                        "|      camera calibration parameters      |",
                        "==========================================="]
        formatstring.extend(
                       ["  parameter |      bounds      |    value  ",
                        "-------------------------------------------"])
        for i,param in enumerate(self.parameter_names):
            formatstring.append(
                "{p:>11} ({bl: 10.3f},{bu: 10.3f}): {pv:>10.3f}".format(
                p=param,pv=getattr(self, param),bl=self.bounds[i][0],
                bu=self.bounds[i][1]))
        return("\n".join(formatstring))



class CameraCalibrationRadialFunction(object):
    """
    Base class for camera calibration radial functions.
    """
    def __init__(self, parameters):
        """
        class constructor
        args:
            parameters(CameraCalibrationParameters): The parameters for the 
                function.
        """
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

    def radiush(self, elevation):
        """ 
        based on the parameters, returns the horizontal radius on the image
        plane given an elevation.
        """
        raise NotImplementedError("radiush has to be implemented!")

    def elevation(self, radiush):
        """ 
        based on the parameters, returns the elevation given a horizontal 
        radius on the image plane.
        """
        raise NotImplementedError("elevation has to be implemented!")

    def __call__(self, elevation=None, radiush=None):
        """ 
        return elevation or radiush depending on what you specify
        """
        if elevation is None and radiush is None:
            raise ValueError("either elevation or radiush have to be defined.")
        elif elevation is None:
            return self.elevation(radiush)
        elif radiush is None:
            return self.radiush(elevation)
        else:
            raise Exception("You should never see this Error...")


class FisheyeEquidistantRadialFunction(CameraCalibrationRadialFunction):
    """
    Ideal equidistant fisheye radial function.
    """
    def radiush(self, elevation):
        p = self.parameters
        return p.r0*elevation

    def elevation(self, radiush):
        p = self.parameters
        return radiush / p.r0
        

class FisheyeEquiangleRadialFunction(CameraCalibrationRadialFunction):
    """
    Ideal equi-angle fisheye radial function.
    """
    def radiush(self, elevation):
        p = self.parameters
        return 2*p.r0*np.tan(elevation/2)

    def elevation(self, radiush):
        return 2 * np.arctan(radiush / (2 * p.r0))

class FisheyeEquiareaRadialFunction(CameraCalibrationRadialFunction):
    """
    Ideal equi-area fisheye radial function.
    """
    def radiush(self, elevation):
        p = self.parameters
        return 2*p.r0*np.sin(elevation/2)

    def elevation(self, radiush):
        p = self.parameters
        return 2 * np.arcsin(radiush / (2 * p.r0))

class FisheyeOrthogonalRadialFunction(CameraCalibrationRadialFunction):
    """
    Ideal orthogonal fisheye radial function.
    """
    def radiush(self, elevation):
        p = self.parameters
        return p.r0*np.sin(elevation)

    def elevation(self, radiush):
        p = self.parameters
        return np.arcsin(radiush / p.r0)

class FisheyePolynomialRadialFunction(CameraCalibrationRadialFunction):
    def __init__(self, parameters, n):
        """
        class constructor
        args:
            parameters(CameraCalibrationParameters): The parameters for the 
                function.
            n (integer): Polynomial degree, i.e. highest exponent.
        """
        super().__init__(parameters)
        self.n = n

    def polynomial(self):
        """
        return the string representation of the polynomial
        """
        polystr = []
        for i in range(self.n):
            polystr.append("r{i}*x**{ie}".format(i=i,ie=i+1))
        return "+".join(polystr)


    def radiush(self, elevation):
        p = self.parameters
        return sum([getattr(p,"r{}".format(i))*elevation**(i+1) \
                                                    for i in range(self.n)])

    def elevation(self, radiush):
        # set up the interpolation
        ele = np.linspace(0,np.pi/2,100) # sequence for elevation
        rh  = self.radiush(ele)
        ele_interpolation = interpolate.interp1d(rh, ele,bounds_error=False)

        
        # return the interpolated values masked where interpolation didn't
        # work
        return np.ma.masked_invalid(ele_interpolation(radiush))



class CameraCalibrationLossFunction(object):
    """
    Class to hold a lossfunction for calibration optimization
    """
    def __init__(self, pixel_coords, sun_coords, radial):
        """
        class constructor.
        args:
            pixel_coords (Coordinates3d): carthesian coordinates
                (y=row,x=col) of sun pixel positions on image.
            sun_coords (Coordinates3d): spherical coordinates
                (azimuth, elevation) of sun positions.
            radial (CameraCalibrationRadialFunction): radial distortion 
                funciton. With the parameter estimate set and the elevation as
                argument, return the image radius in pixel units.
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
        sun_coords.radiush = self.radial.radiush(sun_coords.elevation) 

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
    def __init__(self,shape,method=None):
        """
        class constructor
        args:
            shape (2-tuple of ints): shape of objective image. This is
                important for the optimization parameter bounds and the
                resulting output.
            method (str or None): optimization method. see 
                scipy.optimize.minimize documentation for further information.
                Defaults to None which is not recommended because the standard
                method varies between some scipy versions.
        """
        self.method = method
        self.shape  = shape

    def estimate(self, lossfunc, first_guess):
        """
        fit calibration parameters with respect to a lossfunction given a 
        first guess of parameters.
        args:
            lossfunc (CameraCalibrationLossFunction): the lossfunction
            first_guess (CameraCalibrationParameters): the initial guess
        """
        # set bounds for center_row and center_col
        #first_guess.bounds[0] = (0,int(self.shape[0])) # set row bounds
        #first_guess.bounds[1] = (0,int(self.shape[1])) # set col bounds

        # optimize parameters
        estimate = optimize.minimize(
            lossfunc,
            first_guess.parameters,
            method=self.method,
            bounds=first_guess.bounds
            )

        # create new optimal parameters
        opt_estimate = copy.deepcopy(first_guess) # based on the first guess
        opt_estimate.parameters = estimate.x # set the new parameters

        # create and return a CameraCalibration
        calibration = CameraCalibration(
            parameters = opt_estimate, 
            lossfunc   = lossfunc,
            shape      = self.shape #,fit        = estimate 
            # the fit makes problems at copying/pickling...
            )

        return(calibration)


# calibration class
class CameraCalibration(object):
    """
    class that holds calibration results
    """
    def __init__(self, parameters, lossfunc, shape, fit=None):
        """
        class constructor
        args:
            parameters (CameraCalibrationParameters): calibration parameters
            lossfunc (CameraCalibrationLossFunction): calibration lossfunction
            shape (2-tuple of ints): shape of objective image
            fit (optional[scipy.optimize.Result]): calibration fit
        """

        # copy over attributes
        self.parameters  = parameters
        self.lossfunc    = lossfunc
        self.shape       = shape
        self.fit         = fit

    def create_coordinates(self):
        # create row and col arrays
        row, col = np.mgrid[:self.shape[0],:self.shape[1]]
        # center them
        row = self.shape[0] - row # invert, because y increases to the top
        row = row - self.parameters.center_row
        col = col - self.parameters.center_col

        # create coordinates
        # these coordinates are coordinates on the image!
        plane = coordinates.Coordinates3d(shape=self.shape)
        plane.azimuth_offset    = self.parameters.north_angle
        plane.azimuth_clockwise = self.lossfunc.pixel_coords.azimuth_clockwise
        # first, take x and y as row and col to calculate the azimuth/radiush
        plane.fill(x=col,y=row) # set row and col and calculate azimuth/radiush
    
        # now we can get the elevation based on the horizontal image radius
        elevation = self.lossfunc.radial.elevation(plane.radiush)
        # get the azimuth
        azimuth   = plane.azimuth

        coords = coordinates.Coordinates3d(
            azimuth_offset   =self.lossfunc.pixel_coords.azimuth_offset,
            azimuth_clockwise=True # meteorological azimuth is clockwise
            )
        coords.shape = self.shape # set shape
        # fill the new coordinates with the calculated elevation and azimuth
        coords.fill(elevation=elevation,azimuth=azimuth)
        # return the new coordinates
        return(coords)



