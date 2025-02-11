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
import matplotlib.pyplot as plt

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
    def __init__(self, ang_1=0, ang_2=0, ang_3=0,
                 *radialp):
        """
        class constructor
        args:
            ang_1, ang_2, ang_3(float): euler angles
            radialp (numeric): arbitrary number of radial distortion parameters 
                that have to be sufficient for the radial distortion function.
                They will be named r0 ... rn.
        """
        # set the parameters
        parameters = [ang_1, ang_2, ang_3]
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
            self.bounds[0] = (-np.pi/2,np.pi/2)
            self.bounds[1] = (-np.pi/2,np.pi/2)
            self.bounds[2] = (-np.pi/2,np.pi/2)


    @property
    def parameters(self):
        parms = []
        for p in self.parameter_names:
            parms.append(getattr(self,p))
        return(parms)

    @parameters.setter
    def parameters(self, newparams):
        # create a list of parameter names
        parameter_names = ["ang_1","ang_2","ang_3"]
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
        # interpolate.interp1d instance doesn't like masked arrays, so
        # make sure to convert it to a normal array
        return np.ma.masked_invalid(ele_interpolation(np.asarray(radiush)))


class CameraCalibrationLossFunction(object):
    """
    Class to hold a lossfunction for calibration optimization
    """
    def __init__(self, sun_img, sun_real, imgshape, radial, 
        optimize_projection=True):
        """
        class constructor.
        args:
            sun_img (Coordinates3d): coordinates
                (y=row,x=col) of sun pixel positions on image.
            sun_real (Coordinates3d): coordinates
                (containing azimuth&elevation) of real-world sun positions.
                Since this is a meteorological project, the meteorological
                azimuth definition is expected. That is, looking onto a map
                the azimuth is 0 in the north and increases clockwise.
            imgshape (2-tuple of int): image shape
            radial (CameraCalibrationRadialFunction): radial distortion 
                funciton. With the parameter estimate set and the elevation as
                argument, return the image radius in pixel units.
            optimize_projection (boolean): allow modification of radial
                projection function parameters during optimization? Defaults
                to True.
        returns:
            calibration(CameraCalibrationLossFunction)
        """
        # copy over attributes
        self.sun_img  = sun_img
        self.sun_real = sun_real
        self.radial   = radial
        self.imgshape = imgshape
        self.optimize_projection = optimize_projection

        # Development switches
        self.PLOT    = True # Plot coordinate transformation steps
        self.SKIPPLOT = 100  # how many steps to skip between plots
        self.VERBOSE = True # Print steps


    @staticmethod
    def euler_matrix(a,b,c):
        """
        create a euler turning matrix with three angles.
        Multiply np.array([x,y,z]) to this matrix to get the turned
        coordinates np.array([[x',y',z']]). Follows the zxz convention.
        (see wikipedia)
        args:
            a,b,c (float): turning angles (radian)
        returns:
            matrix (ndarray): turning matrix
        """
        sin = np.sin
        cos = np.cos
        
        mat = np.array([
            [cos(a)*cos(c)-sin(a)*cos(b)*sin(c),
             sin(a)*cos(c)+cos(a)*cos(b)*sin(c),
             sin(b)*sin(c)],
            [-cos(a)*sin(c)-sin(a)*cos(b)*cos(c),
             -sin(a)*sin(c)+cos(a)*cos(b)*cos(c),
             sin(b)*cos(c)],
            [sin(a)*sin(b),-cos(a)*sin(b),cos(b)]
            ])

        return mat

    def project_image_to_real(self,img,real,estimate):
        """
        given 
        args:
            img (Coordinates3d): pixel positions on image (centered
                to optical axis)
            real (Coordinates3d): real position (ele,azi)
            estimate (CameraCalibrationParameters): calibration parameters
        returns:
            projected_sun (Coordinates3d): sun position from image projected
                into real world
        """
        # counter
        try: self.count += 1
        except: self.count = 0

        if self.VERBOSE: logger.debug("Estimate of parameters {}".format(estimate))
        try:    estimate = estimate.parameters
        except: pass
        est_ang_1, est_ang_2, est_ang_3,*radialp = estimate

        if self.VERBOSE: logger.debug((
            "================== begin iteration nr {} "
            "==================").format(self.count))

        # print sun on image coordinates
        if self.VERBOSE: logger.debug(("img: sun coodinates on the image, "
            "centered to image center\n{}").format(img))

        # plot image sun coordinates after radial projection 
        if self.PLOT: 
            if self.VERBOSE: logger.debug("plot base image sun coordinates")
            if self.count % self.SKIPPLOT == 0:
                img.plot()
                plt.gcf().canvas.set_window_title("(1) base img sun")



        #########################################################
        ### project real-world sun coordinates onto the image ###
        #########################################################
        # approximate the sun-on-image elevvation by the horizontal radius
        # on the image with the given radial distortion model
        if self.optimize_projection:
            self.radial.parameters = estimate # set parameters for the radial model
        img.elevation = self.radial.elevation(img.radiush) 

        # plot image sun coordinates after radial projection 
        if self.PLOT: 
            if self.VERBOSE: logger.debug(
                "plot image sun coordinates AFTER radial projection")
            if self.count % self.SKIPPLOT == 0:
                img.plot3d()
                plt.gcf().canvas.set_window_title(
                    "(2) img sun AFTER radial projection")


        # now transform/turn the image sun coordinates to the real sun system
        # --> e.g. image azimuth may turn the other way round (cam looking up)
        ### Adjust the azimuth direction to real sun coordinates ###
        img.change_parameters(
            # the astronomical azimuth is not clockwise on the image
            # so 
            azimuth_clockwise=real.azimuth_clockwise, 
            # adjusted azimuth_offset
            azimuth_offset=real.azimuth_offset,
            # turn by keeping azimuth and radiush
            keep={'azimuth','elevation','radiush'} 
            )

        # print real sun coordinates before projection after projection
        if self.VERBOSE: logger.debug(
          "img: image sun coodinates AFTER projection and turning\n{}".format(img))

        # plot image sun coordinates after projection 
        if self.PLOT: 
            if self.VERBOSE: logger.debug(
                "plot image sun coordinates AFTER radial projection and turning")
            if self.count % self.SKIPPLOT == 0:
                img.plot()
                plt.gcf().canvas.set_window_title(
                    "(3.1) img sun AFTER radial projection and turning")
                img.plot3d()
                plt.gcf().canvas.set_window_title(
                    "(3.2) img sun AFTER radial projection and turning")

        ### now the image sun coordinates are transformed to the real sun
        ### coordinates system
        ### Now adjust the northing
        # create a euler turning matrix
        eul = self.euler_matrix(est_ang_1,est_ang_2,est_ang_3)
        xyz = np.dot(eul,np.array([img.x.ravel(),img.y.ravel(),img.z.ravel()]))
        img.fill(x=xyz[0,:],y=xyz[1,:],z=xyz[2,:])
        # project elevation/azimuth to unit sphere
        img.fill(azimuth   = img.azimuth,
                 elevation = img.elevation,
                 radius    = 1
                 )
        
        # plot image sun coordinates after euler angle adjustion
        if self.PLOT: 
            if self.VERBOSE: logger.debug(
                "plot normalized image sun coordinates AFTER euler turning")
            if self.count % self.SKIPPLOT == 0:
                img.plot3d()
                plt.gcf().canvas.set_window_title(
                    "(4) norm img sun AFTER euler turning")
            
        return img

    def __call__(self, estimate):
        """
        calculate loss function value (residuals) for a given estimate
        args:
            estimate (tuple or CameraCalibrationParameters): estimate
        returns:
            residual = numeric value
        """
        # local copy of coordinates
        sun_img    = copy.deepcopy(self.sun_img) # a copy
        sun_real   = copy.deepcopy(self.sun_real) # a copy

        # project image coordinates to real system
        sun_img = self.project_image_to_real(img=sun_img,real=sun_real,
            estimate=estimate)

        # print real sun coordinates before projection
        if self.VERBOSE: logger.debug(("real: real sun coodinates\n{}").format(sun_real))
        # plot real sun coordinates

        ### project real sun coordinates on sphere 
        sun_real.radius = 1

        if self.PLOT: 
            if self.VERBOSE: logger.debug(
                "plot real sun coordinates on sphere")
            if self.count % self.SKIPPLOT == 0:
                sun_real.plot3d()
                plt.gcf().canvas.set_window_title("real sun on sphere")


        ####################################################################
        ### calculate the residual/difference between projected real and ###
        ### image sun coordinates                                        ###
        ####################################################################
        i = 4
        if i == 1: # optimize x/y matching (WORKS BEST)
            res = np.sqrt((sun_real.x-sun_img.x) ** 2 + \
                      (sun_real.y-sun_img.y) ** 2).mean()
        elif i == 2: # optimize radiush matching
            res = np.sqrt((sun_real.radiush-sun_img.radiush) ** 2).mean()
        elif i == 3: # optimize x/y and radiush matching
            res = np.sqrt((sun_real.x-sun_img.x) ** 2 + \
                          (sun_real.y-sun_img.y) ** 2 + \
                          (sun_real.radiush-sun_img.radiush) ** 2 ).mean()
        elif i == 4: # optimize radius
            res = np.sqrt((sun_real.x-sun_img.x) ** 2 + \
                          (sun_real.y-sun_img.y) ** 2 + \
                          (sun_real.z-sun_img.z) ** 2 ).mean()

        if self.PLOT and self.count % self.SKIPPLOT == 0: plt.show() # show all plots
            
        if res is np.ma.masked:
            raise Exception("Residual is masked. Something is wrong.")

        if self.VERBOSE: logger.debug("Estimate of parameters {}".format(estimate))
        if self.VERBOSE: logger.debug("residual: {res}".format(res=res))
        if self.VERBOSE: logger.debug(("================== end iteration nr {} "
            "=====================").format(self.count))

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

    def estimate(self, lossfunc, first_guess, optimize_projection=True):
        """
        fit calibration parameters with respect to a lossfunction given a 
        first guess of parameters.
        args:
            lossfunc (CameraCalibrationLossFunction): the lossfunction
            first_guess (CameraCalibrationParameters): the initial guess
            optimize_projection (boolean): should the projection be optimized?
                If False, the projection parameters are just fixed from the
                first_guess. Defaults to True.
        """
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

        # no need to save all points, just the metadata
        #lossfunc.sun_img.shape=None
        #lossfunc.sun_real.shape=None

        # create and return a CameraCalibration
        calibration = CameraCalibration(
            parameters = opt_estimate, 
            lossfunc   = lossfunc,
            shape      = self.shape 
            #,fit       = estimate 
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

    def create_coordinates(self, shape=None):
        if shape is None: # no shape specified
            shape = self.shape

        # calculate zoom factor (new = zoom * old)
        zoom = np.unique(np.divide(shape, self.shape))

        # check if specified shape meets aspect ratio
        if zoom.size != 1:
            raise ValueError(" ".join([
                "specified shape {} does not meet calibration",
                "shape {} (aspect ratio does not fit)"]).format(
                shape,self.shape))

        logger.debug(
            "creating new coordinates of shape {}, i.e. zoomed by {}".format(
                shape,zoom))
        
        # create row and col arrays
        row, col = np.mgrid[:shape[0],:shape[1]]
        # center them
        row = shape[0] - (row+1) # invert, because y increases to the top
        row = ( row - shape[0]/2 ) / zoom
        col = ( col - shape[1]/2 ) / zoom

        # create coordinates
        # these coordinates are coordinates on the image!
        plane = coordinates.Coordinates3d(shape=shape)
        plane.azimuth_offset    = self.lossfunc.sun_img.azimuth_offset
        plane.azimuth_clockwise = self.lossfunc.sun_img.azimuth_clockwise
        # first, take x and y as row and col to calculate the azimuth/radiush
        # to prevent the zero-vector from existing, add a small number
        plane.fill(x=col+1e-4,y=row+1e-4) # set row and col and calculate azimuth/radiush
        # now we can get the elevation based on the horizontal image radius
        coords = self.lossfunc.project_image_to_real(img=copy.deepcopy(plane),
            real=self.lossfunc.sun_real,
            estimate=self.parameters)

        coords.fill(
            elevation = np.ma.masked_where(np.logical_or(
                plane.radiush>self.shape[1]/2,coords.elevation<0),
                coords.elevation),
            azimuth = np.ma.masked_where(np.logical_or(
                plane.radiush>self.shape[1]/2,coords.elevation<0),
                coords.azimuth)
            )


        return(coords)



