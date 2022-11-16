
# Set "root" to your top-level directory
root = '/user_data/mmhender/'

# Inside "root" should be the subfolders:
#     "nsd" which includes our pre-processed images from NSD
#     "texturemodel" (main project folder with all our code)
#     "features" (precomputed features extracted for each pRF)
# If you're just trying to load our models that have already been fit (not fitting
# from scratch), then you only need the "texturemodel" folder

project_name = 'texturemodel'

# Set path to the full NSD data repository (this is where the beta weights are stored)
# http://naturalscenesdataset.org/
# This is only needed if you want to fit from scratch
# nsd_path = '/lab_data/tarrlab/common/datasets/NSD'   
nsd_path = ''