
# Set path to your top-level directory
# In here should be the subfolders:
#     "nsd" which includes our pre-processed images and labels
#     "texturemodel" (main project folder with all our code)
#     "features" (precomputed features extracted for each pRF)
root = '/user_data/mmhender/'
project_name = 'texturemodel'

# if using a scratch directory local to the node i'm on, what is its path
# (not used often)
root_localnode = '/scratch/mmhender/'

# Set path to the full NSD data repository (this is where the beta weights are stored)
# http://naturalscenesdataset.org/
nsd_path = '/lab_data/tarrlab/common/datasets/NSD'   

# Path to the COCO API toolbox
# https://github.com/cocodataset/cocoapi
coco_api_path = '/user_data/mmhender/toolboxes/coco_annot'

# Path to where the raw COCO images are stored
# https://cocodataset.org/
coco_ims_path = '/lab_data/tarrlab/common/datasets/COCO'
