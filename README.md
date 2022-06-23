This repo contains code to construct a texture statistics encoding model for fMRI data.

The first step of the model is to extract texture statistics features using a steerable pyramid representation (as in Portilla and Simoncelli 2001). 
Code is included to perform feature extraction from natural images using python/pytorch.  
This code is directly adapted from the matlab code available at: https://github.com/freeman-lab/metamers. 

We also provide code for fitting the texture statistics model to individual voxels using ridge regression.

Any questions/concerns can be directed to mmhender@cmu.edu
