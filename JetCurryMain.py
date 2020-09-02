'''
Copyright 2017, Katie Kosak, Eric Perlman
JETCURRY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

__author__ = "Katie Kosak"
__email__ = "katie.kosak@gmail.com"
__credits__ = ["KunYang Li", "Dr. Eric Perlman", "Dr.Sayali Avachat"]
__status__ = "production"
import imp
import os
from astropy.io import fits
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import JetCurry as jet
import argparse
import glob
import JetCurryGui


# Required modules
MODULES = ['emcee',
           'multiprocessing',
           'scipy',
           'numpy',
           'astropy',
           'matplotlib',
           'math',
           'numpy',
           'pylab',
           'argparse',
           'glob']

'''
Try to import required modules.
If a module can't be imported then print module name and exit
'''
MISSING_MODULES = []

for module in MODULES:
    try:
        imp.find_module(module)
    except BaseException:
        MISSING_MODULES.append(module)

if MISSING_MODULES:
    print('Modules ' + ', '.join(MISSING_MODULES) + ' not installed')
    os.sys.exit()

# Create command line argument parser
PARSER = argparse.ArgumentParser(description="Jet Curry")
PARSER.add_argument('input', help='file or folder name')
PARSER.add_argument('-out_dir', help='output directory path')
ARGS = PARSER.parse_args()

'''
Determine whether input is a single file or directory
Create list of FITS files for processing
'''
FILES = []
if os.path.isfile(ARGS.input):
    try:
        fits.PrimaryHDU.readfrom(ARGS.input)
        FILES.append(ARGS.input)
    except BaseException:
        print('%s is not a valid FITS file!' % ARGS.input)
        os.sys.exit()
else:
    if os.path.exists(ARGS.input):
        for file in glob.glob(ARGS.input + '/*.fits'):
            try:
                fits.PrimaryHDU.readfrom(file)
                FILES.append(file)
            except BaseException:
                print('%s is not a valid FITS file!' % file)
    else:
        print('Input directory does not exist!')

# Create output directory if it doesn't exitst
if ARGS.out_dir is not None:
    if not os.path.exists(ARGS.out_dir):
        try:
            os.makedirs(ARGS.out_dir)
        except BaseException:
            print('%s does not exist and cannot be created' % ARGS.out_dir)
else:
    ARGS.out_dir = os.getcwd()

if ARGS.out_dir[-1] == '/':
    OUTPUT_DIRECTORY_DEFAULT = ARGS.out_dir
else:
    OUTPUT_DIRECTORY_DEFAULT = ARGS.out_dir + '/'

np.seterr(all='ignore')
S = []
ETA = []
# Line of Sight (radians)
THETA = 0.261799388


for file in FILES:
    filename = os.path.splitext(file)[0]
    filename = os.path.basename(filename)
    OUTPUT_DIRECTORY = OUTPUT_DIRECTORY_DEFAULT + filename + '/'
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    curry = JetCurryGui.JetCurryGui(file)
    UPSTREAM_BOUNDS = np.array(
        [curry.x_start_variable.get(), curry.y_start_variable.get()])
    DOWNSTREAM_BOUNDS = np.array(
        [curry.x_end_variable.get(), curry.y_end_variable.get()])
    NUMBER_OF_POINTS = DOWNSTREAM_BOUNDS[0] - UPSTREAM_BOUNDS[0]

    pixel_min = np.nanmin(curry.fits_data)
    pixel_max = np.nanmax(curry.fits_data)

    # Square Root Scaling for fits image
    data = jet.imagesqrt(curry.fits_data, pixel_min, pixel_max)
    plt.imshow(data)

    # Go column by column to calculate the max flux for each column
    x, y, x_smooth, y_smooth, intensity_max = jet.Find_MaxFlux(
        data, UPSTREAM_BOUNDS, DOWNSTREAM_BOUNDS, NUMBER_OF_POINTS)
    
    # Calculate the s and eta values
    # s,eta, x_smooth,y_smooth values will
    # be stored in parameters.txt file

    S, ETA = jet.Calculate_s_and_eta(
        x_smooth, y_smooth, UPSTREAM_BOUNDS, OUTPUT_DIRECTORY, filename)

    # Run the First MCMC Trial in Parallel
    jet.MCMC1_Parallel(S, ETA, THETA, OUTPUT_DIRECTORY, filename)
    jet.MCMC2_Parallel(S, ETA, THETA, OUTPUT_DIRECTORY, filename)
    
    # Run Simulated Annealing to guarantee Real Solution
    jet.Annealing1_Parallel(S, ETA, THETA, OUTPUT_DIRECTORY, filename)
    jet.Annealing2_Parallel(S, ETA, THETA, OUTPUT_DIRECTORY, filename)
    
    x_coordinates, y_coordinates, z_coordinates = jet.Convert_Results_Cartesian(S, ETA, THETA, OUTPUT_DIRECTORY, filename)
    
    #Square Root Scaling for fits image
    data = jet.imagesqrt(curry.fits_data, pixel_min, pixel_max)
    
    # Go column by column to calculate the max flux for each column
    x, y, x_smooth, y_smooth, intensity_max = jet.Find_MaxFlux(
        data, UPSTREAM_BOUNDS, DOWNSTREAM_BOUNDS, NUMBER_OF_POINTS)
    
    with open(OUTPUT_DIRECTORY + filename + '_flux_maxima_coords.txt', 'w') as file:
        for i in range(len(x)):
            file.write(str(x[i]) + '\t' + str(y[i]) 
                       + '\t' + str(x_smooth[i]) + '\t' 
                       + str(y_smooth[i]) + '\t' + str(intensity_max[i]) + '\n')
            
    #To save the squares root scaling fits image (from JetCurry Code):
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(OUTPUT_DIRECTORY + filename + '_data.fits') 
    
    # Plot the Max Flux over the Image
    plt.contour(data, 10, cmap='gray')
    plt.savefig(OUTPUT_DIRECTORY + filename + '_contour.png')
    plt.scatter(x_smooth, y_smooth, c='b')
    plt.scatter(x, y, c='b')
    plt.title('Outline of Jet Stream')
    ax = plt.gca()
    ax.invert_yaxis()
    
    #plt.show()
    plt.savefig(OUTPUT_DIRECTORY + filename + '_contour.png')
    plt.clf()
    
    #Reads the contents of the Cartesian Coordinates File and loads the contents 
    #NumPy arrays:
    file_name = 'KnotD_Radio_Cartesian_Coordinates.txt'
    cart_coords_file = os.path.join(current_working_directory, file_name)

    x_unsorted, y_unsorted, z_unsorted = jet.read_Cartesian_Coordinates_File(cart_coords_file)

    #PCA:
    coords_cal, PCA_line_coords, t = jet.PCA(x_unsorted,y_unsorted,z_unsorted)

    #PCA Cartesian coordinates:
    x_pca_unsorted = coords_cal[:,0]
    y_pca_unsorted = coords_cal[:,1]
    z_pca_unsorted = coords_cal[:,2]

    '''
    #To bin and sort the Cartesian coordinates based on user-defined number of subintervals:
    #Number of sunbintervals:
    N = 10
    x_binned, y_binned, z_binned, x_pca_binned, y_pca_binned, z_pca_binned, t_binned = jet.BinAndSort(
    x_unsorted, y_unsorted, z_unsorted, x_pca_unsorted, y_pca_unsorted, z_pca_unsorted, t, N)
    '''

    # Plot the Results on Image
    plt.scatter(x_coordinates, y_coordinates, c='y')
    plt.scatter(x_smooth, y_smooth, c='r')
    ax = plt.gca()
    ax.invert_yaxis()
    # plt.show()
    plt.savefig(OUTPUT_DIRECTORY + filename + '_sim.png')
    plt.clf()
    
    #To Plot Spline Fit of Flux Maxima:
    fig = plt.figure(figsize=(10,6))
    ax = plt.gca(projection='3d')

    # To load the data for Plotting the Spline Fit of Max Flux over the Image:
    # By using zdir='y', the y value of these points is fixed to the zs value 0
    # and the (x,y) points are plotted on the x and z axes.
    # create a 35 x 99 mesh
    xx, zz = np.meshgrid(np.linspace(0,99,99), np.linspace(0,35,35))
    # create vertices for a rotated mesh (3D rotation matrix)
    X =  xx 
    Z =  zz
    Y =  10*np.ones(X.shape)

    #To plot contours as a Background:
    #Name of the FITS File with file path:
    file_name = "KnotD_Radio_data.fits"
    image_file = current_working_directory + '/' + file_name
    data = fits.getdata(image_file, ext=0)

    #To plot contours as a Background:
    ax.contourf(X, data, Z, 50, alpha = 0.5, zdir = 'y', cmap = cm.spectral, linewidths= 0.5)
    ax.contour(X, data, Z, 10, zdir = 'y', c = 'k', linewidths= 0.8)

    #To plot the most probable solutions of from Jet Curry (Red Points) >> 2D Projection:
    ax.scatter(x_unsorted, y_unsorted, zs = 0, zdir='y', 
               color = 'red', 
               label = "Most Probable Solutions in Cartesian Coordinates", s = 2)

    #The Most Probable Solutions of the non-linear parametrized eqautions:
    ax.scatter3D(x_unsorted, y_unsorted, z_unsorted, zdir='y', color = 'red', s = 5)

    #To Illustrate the Most Probable Coordinates: (Lines Connecting the Projections):
    #Red Dotted Lioned Connecting Most Probable 3D Corrdinates with 2D Projections
    for i,j,k, in zip(x_unsorted, y_unsorted, z_unsorted):
        ax.plot([i,i],[k,0],
                [j,j], 
                color = 'r',
                marker = ".", linewidth= 0.3)

    #To plot the preferred Direction:
    #To plot the calculated coordinates (Calculated using PCA):
    ax.scatter3D(coords_cal[:,0],coords_cal[:,1],coords_cal[:,2], 
                 zdir='y', marker = "+", color = 'blue', 
                 s = 10, linewidths=0.2, 
                 label = "Projected Coordinates (using PCA)")

    #To plot the lines passing through all PCA coordinates:
    ax.plot(PCA_line_coords[:,0], PCA_line_coords[:,1], PCA_line_coords[:,2], 
            zdir='y', 
            label = "Preferred Direction: Direction generated by applying PCA")  

    #To Plot Origin:
    #The Origin coordinates represent the (x,y) Cartesian 
    #for coordinates at the start of the stream
    x_origin = 22
    y_origin = 21

    ax.scatter(x_origin, y_origin, zdir='y', 
               s = 90, marker='x', c ="k", 
               label = "Origin")

    #To Plot Line of Sight (LOS):
    viewing_angle = 15 #LOS in Degrees
    length = 40 #Length of LOS

    #To find the end point:
    endx = x_origin + length * np.sin(np.radians(viewing_angle))
    endy = y_origin + length * np.cos(np.radians(viewing_angle))

    #plot the points
    ax.plot([x_origin, endx], [0, endy], [y_origin, y_origin], 
            c = "g", 
            marker = "<", 
            label = "Line of Sight (with respect to Z-axis): Viewing Angle: %s Deg" %(viewing_angle))

    #To Plot "z" axis: Here, "z" refers to the one in jet curry paper:
    ax.plot([22, 22],[0, 60], [21,21], c = "k", marker = "<", label = "Z Axis") 

    #3-D Plot Visualization Parameters:
    ax.zaxis.set_rotate_label(False) 
    ax.set_xlabel('X: $\longrightarrow$')
    ax.set_ylabel('$\longleftarrow$: Z')
    ax.set_zlabel('Y', rotation = 0)

    ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, 0.03), 
              fancybox = True, ncol = 3, borderaxespad = 0.)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.set_zlim(0, 40)

    ax.invert_yaxis()
