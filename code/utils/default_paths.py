# this file sets relative paths to folders of interest for our project
# see path_defs.py to change the root directory/absolute paths

import os
import path_defs

nsd_path = path_defs.nsd_path
nsd_root = nsd_path 
root = path_defs.root
project_name = path_defs.project_name

# Where we are keeping the preprocessed NSD stimuli/labeling data
stim_root = os.path.join(root, 'nsd','stimuli')    
stim_labels_root = os.path.join(root, 'nsd','labels')    
nsd_rois_root = os.path.join(root, 'nsd', 'rois')

# Where to save model fits
save_fits_path = os.path.join(root, project_name, 'model_fits')

# Where to save any figures we make
fig_path = os.path.join(root, project_name, 'figures')

# Path where gabor model features will be saved
gabor_texture_feat_path = os.path.join(root, 'features','gabor_texture')

# Path where texture model features will be saved
pyramid_texture_feat_path = os.path.join(root, 'features', 'pyramid_texture')

# Path where AlexNet features will be saved
alexnet_feat_path = os.path.join(root, 'features','alexnet')

# Path where gist model features will be saved
gist_feat_path = os.path.join(root, 'features','gist')

# Where the raw NSD beta weights are located
beta_root = os.path.join(nsd_root,'nsddata_betas','ppdata')
