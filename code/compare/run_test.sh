#!/bin/bash

# To use this: first need to clone "metamers" and "matlabPyrTools" into the same directory as this 
# file (texturemodel/code/compare/).
# From https://github.com/LabForComputationalVision/matlabPyrTools
# and https://github.com/freeman-lab/metamers
# Those folders will be added to path automatically in matlab.

# It will print out some numerical values for the different feature types, which you can 
# then compare to the values output by the jupyter notebook in this same folder
# (which runs the python code)

module load matlab-9.7

matlab -nodisplay -nodesktop -nosplash -r "run_test; exit"
