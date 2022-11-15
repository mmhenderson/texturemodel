import os
from utils import default_paths
 
# hard coded names of previously fit models that need to be loaded for later steps of fitting

texture_fit_paths = ['S01/texture_pyramid_ridge_4ori_4sf_pcaHL_fit_pRFs/Jul-06-2022_1646_59/all_fit_params.npy', \
                 'S02/texture_pyramid_ridge_4ori_4sf_pcaHL_fit_pRFs/Jul-06-2022_1855_20/all_fit_params.npy',\
                 'S03/texture_pyramid_ridge_4ori_4sf_pcaHL_fit_pRFs/Jul-06-2022_2106_02/all_fit_params.npy', \
                 'S04/texture_pyramid_ridge_4ori_4sf_pcaHL_fit_pRFs/Jul-06-2022_2319_30/all_fit_params.npy', \
                 'S05/texture_pyramid_ridge_4ori_4sf_pcaHL_fit_pRFs/Jul-07-2022_0114_01/all_fit_params.npy', \
                 'S06/texture_pyramid_ridge_4ori_4sf_pcaHL_fit_pRFs/Jul-07-2022_0328_12/all_fit_params.npy', \
                 'S07/texture_pyramid_ridge_4ori_4sf_pcaHL_fit_pRFs/Jul-07-2022_0553_54/all_fit_params.npy', \
                 'S08/texture_pyramid_ridge_4ori_4sf_pcaHL_fit_pRFs/Jul-07-2022_0741_10/all_fit_params.npy']

texture_fit_paths = [os.path.join(default_paths.save_fits_path, aa) for aa in texture_fit_paths]