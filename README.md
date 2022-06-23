Code to construct/analyze a texture statistics encoding model for fMRI data.

The first step of the model is to extract texture statistics features using a steerable pyramid representation (as in Portilla and Simoncelli 2001). 
Code is included to perform feature extraction from natural images. This code requires PyTorch as well as PyrTools (https://pyrtools.readthedocs.io/en/latest/).

Our feature extraction code is directly adapted from the matlab code available at: https://github.com/freeman-lab/metamers. 
We provide some functions to do a direct comparison of our method with the Freeman method, inside /code/compare/. 
Running this comparison will require the following repositories:
  - https://github.com/freeman-lab/metamers
  - https://github.com/LabForComputationalVision/matlabPyrTools

We also provide code for fitting the texture statistics model to individual voxels using ridge regression.

Any questions/concerns can be directed to mmhender@cmu.edu
