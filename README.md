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
  <li>Use the jupyter notebooks inside "notebooks" to plot the results of model fitting, and reproduce all our figures.
  <ul>
    <li> Some of the plots require PyCortex, which you can download here:
    <li> https://pypi.org/project/pycortex/
  </ul>
  
</ol>

### Feature extraction:
The first step of the fitting procedure is to extract texture statistics features using a steerable pyramid representation. Our code is adapted from the Matlab code available at: https://github.com/freeman-lab/metamers. 

Running the feature extraction code requires PyTorch as well as PyrTools (https://pyrtools.readthedocs.io/en/latest/). Using a GPU is recommended for speed. 

See "code/run/extract_texture_feats.sh" for an example of how to run the feature extraction code (adjust the paths in this script for your local filesystem).

If you don't want to extract the features from scratch, you can download our pre-computed features at: https://osf.io/8fsnx/

### Model fitting:
See the scripts in "code/run/fit..." for examples of how to run fitting code (adjust the paths in these scripts for your local filesystem).

To fit the models, you'll need to first download the NSD dataset, and run the feature extraction code (or download the features from OSF).

### Other notes:

Any questions/concerns can be directed to mmhender@cmu.edu
