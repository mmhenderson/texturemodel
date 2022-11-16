Code to construct/analyze a texture statistics encoding model for fMRI data.
See our preprint at: https://www.biorxiv.org/content/10.1101/2022.09.23.509292v1

### Setup instructions:
<ol>
  <li>Clone the repository: 
  <ul>
    <li> git clone git@github.com:mmhenderson/texturemodel
  </ul>
  <li>Edit "root" in path_defs.py to reflect the name of the folder into which you cloned this repository.
  <li>If using our pre-computed model fits: download the data from OSF
  <ul>
    <li> https://osf.io/8fsnx/
    <li> After unzipping, you should have a folder "model_fits", which can be placed inside "texturemodel".
  </ul>
  <li>If fitting from scratch: access the fMRI dataset (NSD) and images here:
  <ul>
    <li> http://naturalscenesdataset.org/
    <li> Update path_defs.py to reflect the path where NSD is downloaded.
    <li> Use "/code/run/prep_images.sh" to prepare the NSD images for feature extraction pipeline.
  </ul>
</ol>

### Feature extraction:
The first step of the model is to extract texture statistics features using a steerable pyramid representation (as in Portilla and Simoncelli 2001). 
Code is included to perform feature extraction from natural images. This code requires PyTorch as well as PyrTools (https://pyrtools.readthedocs.io/en/latest/).

Our feature extraction code is directly adapted from the matlab code available at: https://github.com/freeman-lab/metamers. 
We provide some functions to do a direct comparison of our method with the Freeman method, in /code/compare/. 
Running this comparison will require the following repositories:
  - https://github.com/freeman-lab/metamers
  - https://github.com/LabForComputationalVision/matlabPyrTools

### Model fitting:

Any questions/concerns can be directed to mmhender@cmu.edu
