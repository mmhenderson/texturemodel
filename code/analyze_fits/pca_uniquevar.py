import os
import numpy as np

from sklearn import decomposition

from utils import default_paths, roi_utils, nsd_utils, stats_utils
from plotting import load_fits, summary_plots

def do_pca_uniquevar():
    
    subjects = np.arange(1,9)
    n_subjects = len(subjects)

    fitting_type = 'texture_pyramid_ridge_4ori_4sf_pcaHL_allsubsets_use_texture_pRFs'

    out = [load_fits.load_fit_results(subject=ss, fitting_type=fitting_type, \
                                      n_from_end=0, verbose=False) for ss in subjects]
    fig_save_folder=None

    # shuffled model fits, to compute significance of model fit each ROI.
    fitting_type_shuffle = 'texture_pyramid_ridge_4ori_4sf_pcaHL_use_texture_pRFs_permutation_test'
    out_shuff = [load_fits.load_fit_results(subject=ss, fitting_type=fitting_type_shuffle, \
                                      n_from_end=0, verbose=False) \
                                        for ss in subjects]

    # Process results of permutation test, compute p-values
    pvals_singlevoxels_modelsig = [[] for si in range(n_subjects)]

    for si in range(n_subjects):

        # stats for single voxels
        r2_real = out[si]['val_r2']
        r2_shuff = out_shuff[si]['val_r2']

        # for how many of the shuffle iterations did shuffle-R2 exceed real-R2?
        p = np.mean(r2_real[:,0,None]<=r2_shuff[:,0,:], axis=1)
        _,pvals_fdr = stats_utils.fdr_keepshape(p, alpha=0.01, \
                                                       method='poscorr')

        pvals_singlevoxels_modelsig[si] = pvals_fdr

    # compute unique variance each feature type
    # concatenated across all subjects
    ff_inds = np.arange(1,11)
    
    unique_var_vals_all = []
    vox2use_all = []
    subject_inds = []

    for si, ss in enumerate(subjects):

        val_r2 = out[si]['val_r2']        
        var_expl = val_r2[:,0:1] - val_r2[:,ff_inds] 
        model_sig = pvals_singlevoxels_modelsig[si]<0.01
        abv_thresh = summary_plots.get_noise_ceiling(out[si])>0.01
        vox2use = model_sig & abv_thresh 
        # use voxels with good noise ceiling, and passing permutation test
        
        unique_var_vals = var_expl[vox2use,:]
        
        unique_var_vals_all += [unique_var_vals]
        vox2use_all += [vox2use]
        subject_inds += [si*np.ones(np.sum(vox2use),)]

    unique_var_vals_all = np.concatenate(unique_var_vals_all, axis=0)
    subject_inds = np.concatenate(subject_inds, axis=0)
   
    # now do PCA across all the values
    values = unique_var_vals_all.T

    n_features_actual = values.shape[0]
    n_trials = values.shape[1]

    n_comp = np.min([n_features_actual, n_trials])

    pca = decomposition.PCA(n_components = n_comp, copy=True)
    scores = pca.fit_transform(values)           

    # weights are [PCs x voxels]
    wts = pca.components_
    ev = pca.explained_variance_
    ev = ev/np.sum(ev)*100
    pre_mean = pca.mean_

    wts_eachsubj = [wts[:,subject_inds==si].T for si in range(n_subjects)]

    filename_save = os.path.join(default_paths.root, default_paths.project_name, \
                                 'pca_uniquevar', 'pca_wts.npy')

    np.save(filename_save, {'wts_eachsubj': wts_eachsubj, \
                            'vox2use_all': vox2use_all, 
                            'scores': scores, \
                            'ev': ev})
    

def get_texture_pc_activations(subjects=[1,2], sessions=[0]):

    # load the weights for PCA (performed on unique variance for each texture model feature subtype)
    filename = os.path.join(default_paths.root, default_paths.project_name, \
                            'pca_uniquevar', 'pca_wts.npy')
    pca_wts = np.load(filename, allow_pickle=True).item()
    wts = pca_wts['wts_eachsubj']
    vox2use = pca_wts['vox2use_all']

    n_vox_each = [np.sum(vv) for vv in vox2use]
    subject_inds_full = np.concatenate([np.tile(np.array([si]), [nv,]) \
                                    for si, nv in enumerate(n_vox_each)], axis=0)

    n_total_vox = np.sum(n_vox_each)
    
    n_subjects = len(subjects)

    # load voxel data, all subjects
    val_images = 1000;
    voxel_data_allsubs = np.zeros((val_images, n_total_vox))
    is_defined = np.zeros((val_images, n_subjects),dtype=bool)

    for si, ss in enumerate(subjects):

        voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = \
                                        roi_utils.get_voxel_roi_info(subject=ss)

        voxel_data, image_order, val_inds, holdout_inds, session_inds =\
            nsd_utils.get_data_splits(subject=ss, sessions=sessions, voxel_mask=voxel_mask, 
                                     average_image_reps=True)

        # taking just validation set data, because these are the overlapping images
        val_voxel_data = voxel_data[val_inds,:]
        val_image_inds = image_order[val_inds]

        # gathering only voxels that were used in the PCA (had good texture model fits)
        voxel_data_use = val_voxel_data[:,vox2use[si]]
        
        # which images are defined for this subject?
        is_defined[val_image_inds,si] = True

        # add data into big array
        voxel_data_thissub = voxel_data_allsubs[:,subject_inds_full==si] 

        voxel_data_thissub[val_image_inds,:] = voxel_data_use

        voxel_data_allsubs[:,subject_inds_full==si] = voxel_data_thissub

    
    voxel_data_subjuse = voxel_data_allsubs[:, np.isin(subject_inds_full, (np.array(subjects)-1))]

    # identify which images are present for all subjects
    overlapping_inds = np.all(is_defined, axis=1)

    image_inds_overlappingsubs = np.where(overlapping_inds)[0]

    voxel_data_overlappingsubs = voxel_data_subjuse[overlapping_inds,:]
    
    # subtracting off the mean for each voxel over trials
    # voxel_data_overlappingsubs = voxel_data_overlappingsubs - np.mean(voxel_data_overlappingsubs, axis=0, keepdims=True)
    # voxel_data_overlappingsubs = voxel_data_overlappingsubs - np.mean(voxel_data_overlappingsubs, axis=1, keepdims=True)

    # pca weights for all subjects
    wts_allsubs = np.concatenate([wts[ss-1] for ss in subjects], axis=0)
    
    # project to pca space, proj is images x PCs
    proj = voxel_data_overlappingsubs @ wts_allsubs
    
    filename_save = os.path.join(default_paths.root, 'texturemodel', 'pca_projected_activs.npy')
    
    np.save(filename_save, {'proj': proj, 
                           'image_inds': image_inds_overlappingsubs})
    
if __name__ == '__main__':
    
    subjects=np.arange(1,9)
    sessions=np.arange(0,40)
    
    get_texture_pc_activations(subjects=subjects, sessions=sessions)
    