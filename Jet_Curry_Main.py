# Copyright 2017, Katie Kosak, Eric Perlman
#  JETCURRY is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

__author__="Katie Kosak"
__email__="katie.kosak@gmail.com"
__credits__=["KunYang Li", "Dr. Eric Perlman", "Dr.Sayali Avachat"]
__status__="production"
from scipy import array
import pyfits
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import *
import Jet_Curry as jet
from scipy.interpolate import spline
######################## Notes #######################################
#### Make sure in code to have the following python packages installed
#### This can be done with pip install in terminal
#### emcee, corner, multiprocessing, scipy, numpy, pyfits,matplotlib
#### math, numpy, pylab
####
#### A Better Description for the code is included in ReadME on github

#### Line's with 4 #'s is instructions/comments
#### Line's with 1 # is optional/alternative code to use if User wishes
#######################################################################

#### Open the image from fits file 
#### Obtain image parameters
#### Image will read in a NASA standard fits file
#### If you do not know the pixel position of the upstream, downstream bounds,
#### You can either plot the image with the commented out code and manually 
#### include it in the code. 
#filename='File_Name.fits'
#file1=pyfits.getdata(filename)
#plt.imshow(file1)
#plt.show()
#### Or you can use ginput below
#filename='AB.fits'
#file1=pyfits.getdata(filename)
#plt.imshow(file1)
#Bounds=plt.ginput(2) 
####Make sure the mouse clicks Upstream -> Downstream
####then replace Bounds lines of code to:
####Comment out the other Upstream/Downstream Bounds if you do. 
#Upstream_Bounds=Bounds[0]
#Downstream_Bounds=Bounds[1]
np.seterr(all='ignore')
s=[]
eta=[]
#### Line of Sight (radians)
theta=0.261799388 
nameofknots='KnotD_Radio'
filename=nameofknots+'.fits'
#### Position of end point on image for fit
### Note: Image must be input coordinates of (y,x)
Downstream_Bounds=np.array([193,36]) 
#### Position of starting point on image for fit
### Note: Image must be input coordinates of (y,x)
Upstream_Bounds=np.array([6,36]) 
number_of_points=Downstream_Bounds[1]-Upstream_Bounds[1]

#### If you do not know the pixel position of the upstream, downstream
#### See Note Above.
nameofknots='AB'
filename=nameofknots+'.fits'

####  Obtain Information from the Image/Show the Image
file1=pyfits.getdata(filename)
pixel_min=np.nanmin(file1)
pixel_maxima=np.nanmax(file1)
#### Square Root Scaling for fits image
file1=jet.imagesqrt(file1,pixel_min,pixel_maxima) 
plt.imshow(file1)

#### Go column by column to calculate the max flux for each column
x,y,x_smooth,y_smooth,intensity_max=jet.Find_MaxFlux(file1,Upstream_Bounds,Downstream_Bounds,number_of_points)

#### Plot the Max Flux over the Image
plt.contour(file1,10, cmap='gray')
plt.scatter(x_smooth,y_smooth,c='b')
plt.scatter(x,y,c='b')
plt.title('Outline of Jet Stream')
ax = plt.gca()
ax.invert_yaxis()
plt.show()

#### Calculate the s and eta values
#### s,eta, x_smooth,y_smooth values will 
#### be stored in parameters.txt file

s,eta=jet.Calculate_s_and_eta(x_smooth,y_smooth,Upstream_Bounds)
#### Run the First MCMC Trial in Parallel
jet.MCMC1_Parallel(s,eta,theta)
jet.MCMC2_Parallel(s,eta,theta)
#### Run Simulated Annealing to guarantee Real Solution
jet.Annealing1_Parallel(s,eta,theta)
jet.Annealing2_Parallel(s,eta,theta) 
x_coordinates,y_coordinates,z_coordinates=jet.Convert_Results_Cartesian(s,eta,theta)

#### Plot the Results on Image   
plt.scatter(x_coordinates,y_coordinates,c='y')
plt.scatter(x_smooth,y_smooth,c='r')
ax = plt.gca()
ax.invert_yaxis()
plt.show()
