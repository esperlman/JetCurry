JetCurry, Copyright 2017. Katie Kosak. Eric Perlman. 
Copyright Details: GPL.txt 


Last Edited: March 2017. Questions: Contact Katie Kosak, katie.kosak@gmail.com

Python code that models the 3D geometry of AGN jets

Make sure the files are in the same directory.

Jet_Curry has the functions for performing the calculation from 2D to 3D. There is no need to change this code to perform calculations. It is best not to mess around with this code.

Jet_Curry_Main is the code that the user will use to define the fits file, define the core location, etc.

To call in the tools used from Jet_Curry, the following command is used:

import Jet_Curry as jet

To use the functions necessary for the calculations, use the following:

jet.Find_MaxFlux(file1,Upstream_Bounds,number_of_points)

To execute the code, execute the Jet_Curry_Main.py code as a normal Python code. If the Jet_Curry.pyc file is in the same directory as Jet_Curry_Main, the functions will be automatically imported.

where jet tells the code to pull the function from the code Jet_Curry. Find_MaxFlux is the function in Jet_Curry being called.

For the functions in Jet_Curry, all arguments must be included as the example.

To use the Jet_Curry Visualization with VPython, Recommend using Jupyter's IPython Notebooks.

