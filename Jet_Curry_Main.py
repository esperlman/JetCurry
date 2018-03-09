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
import Jet_Curry as jet
import argparse
import glob
import Jet_Curry_Gui

# Required modules
MODULES = ['emcee',
		   'corner',
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
missingModules = []

for module in MODULES:
	try:
		imp.find_module(module)
	except:
		missingModules.append(module)

if missingModules:
	print('Modules ' + ', '.join(missingModules) + ' not installed')
	os.sys.exit()

# Create command line argument parser
parser = argparse.ArgumentParser(description="Jet Curry")
parser.add_argument('input', help='file or folder name')
parser.add_argument('-out_dir', help='output directory path')
args = parser.parse_args()

'''
Determine whether input is a single file or directory
Create list of FITS files for processing
'''
files = []
if os.path.isfile(args.input):
	try:
		fits.PrimaryHDU.readfrom(args.input)
		files.append(args.input)
	except:
		print('%s is not a valid FITS file!' % args.input)
		os.sys.exit()
else:
	if os.path.exists(args.input):
		for file in glob.glob(args.input + '/*.fits'):
			try:
				fits.PrimaryHDU.readfrom(file)
				files.append(file)
			except:
				print('%s is not a valid FITS file!' % file)
	else:
		print('Input directory does not exist!')

# Create output directory if it doesn't exitst
if args.out_dir is not None:
	if not os.path.exists(args.out_dir):
		try:
			os.makedirs(args.out_dir)
		except BaseException:
			print('%s does not exist and cannot be created' % args.out_dir)
else:
	args.out_dir = os.getcwd()

output_directory = args.out_dir + '/'

np.seterr(all='ignore')
s = []
eta = []
# Line of Sight (radians)
THETA = 0.261799388


for file in files:
	filename = os.path.splitext(file)[0]
	filename = os.path.basename(filename)
	output_directory = output_directory + filename + '/'
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)

	curry = Jet_Curry_Gui.Jet_Curry_Gui(file)
	UPSTREAM_BOUNDS = np.array([curry.xStartVariable.get(), curry.yStartVariable.get()])
	DOWNSTREAM_BOUNDS = np.array([curry.xEndVariable.get(), curry.yEndVariable.get()])
	NUMBER_OF_POINTS = DOWNSTREAM_BOUNDS[0] - UPSTREAM_BOUNDS[0]

	pixel_min = np.nanmin(curry.fits_data)
	pixel_max = np.nanmax(curry.fits_data)

	# Square Root Scaling for fits image
	data = jet.imagesqrt(curry.fits_data, pixel_min, pixel_max)
	plt.imshow(data)

	# Go column by column to calculate the max flux for each column
	x, y, x_smooth, y_smooth, intensity_max = jet.Find_MaxFlux(data, UPSTREAM_BOUNDS, DOWNSTREAM_BOUNDS, NUMBER_OF_POINTS)

	# Plot the Max Flux over the Image
	plt.contour(data, 10, cmap='gray')
	plt.scatter(x_smooth, y_smooth, c='b')
	plt.scatter(x, y, c='b')
	plt.title('Outline of Jet Stream')
	ax = plt.gca()
	ax.invert_yaxis()
	plt.show()
	plt.savefig(output_directory + filename + '_contour.png')

	# Calculate the s and eta values
	# s,eta, x_smooth,y_smooth values will
	# be stored in parameters.txt file

	s, eta = jet.Calculate_s_and_eta(x_smooth, y_smooth, UPSTREAM_BOUNDS, output_directory, filename)
	# Run the First MCMC Trial in Parallel
	jet.MCMC1_Parallel(s, eta, THETA, output_directory, filename)
	jet.MCMC2_Parallel(s, eta, THETA, output_directory, filename)
	# Run Simulated Annealing to guarantee Real Solution
	jet.Annealing1_Parallel(s, eta, THETA, output_directory, filename)
	jet.Annealing2_Parallel(s, eta, THETA, output_directory, filename)
	x_coordinates, y_coordinates, z_coordinates = jet.Convert_Results_Cartesian(
	    s, eta, THETA, output_directory, filename)

	# Plot the Results on Image
	plt.scatter(x_coordinates, y_coordinates, c='y')
	plt.scatter(x_smooth, y_smooth, c='r')
	ax = plt.gca()
	ax.invert_yaxis()
	plt.show()
