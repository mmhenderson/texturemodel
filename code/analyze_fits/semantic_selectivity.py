import numpy as np
from utils import stats_utils

def get_semantic_discrim(best_prf_inds, labels_all, unique_labels_each, val_voxel_data_pred, \
                         trials_use_each_prf = None, debug=False):
   
    """
    Measure how well voxels' predicted responses distinguish between image patches with 
    different semantic content.
    """
    
    n_voxels = val_voxel_data_pred.shape[1]

    n_sem_axes = labels_all.shape[1]
    sem_discrim_each_axis = np.zeros((n_voxels, n_sem_axes))
    sem_corr_each_axis = np.zeros((n_voxels, n_sem_axes))
    
    max_categ = np.max([len(un) for un in unique_labels_each])
    n_samp_each_axis = np.zeros((n_voxels, n_sem_axes, max_categ),dtype=np.float32)
    mean_each_sem_level = np.zeros((n_voxels, n_sem_axes, max_categ),dtype=np.float32)
 
    for vv in range(n_voxels):
        
        if debug and (vv>1):
            continue
        print('computing semantic discriminability for voxel %d of %d, prf %d\n'%(vv, n_voxels, best_prf_inds[vv,0]))
        
        resp = val_voxel_data_pred[:,vv,0]
        
        if trials_use_each_prf is not None:
            # select subset of trials to work with
            trials_use = trials_use_each_prf[:,best_prf_inds[vv,0]]
            resp = resp[trials_use]
            labels_use = labels_all[trials_use,:,:]
            if np.sum(trials_use)==0:
                print('voxel %d: no trials are included here, skipping it'%vv)
                sem_discrim_each_axis[vv,:] = np.nan
                sem_corr_each_axis[vv,:] = np.nan
                continue
        else:
            labels_use = labels_all
       
        for aa in range(n_sem_axes):
            
            labels = labels_use[:,aa,best_prf_inds[vv,0]]
            
            inds2use = ~np.isnan(labels)
            
            unique_labels_actual = np.unique(labels[inds2use])
            
            if np.all(np.isin(unique_labels_each[aa], unique_labels_actual)):
                
                # separate trials into those with the different labels for this semantic axis.
                group_inds = [(labels==ll) & inds2use for ll in unique_labels_actual]
                groups = [resp[gi] for gi in group_inds]
                
                if len(unique_labels_actual)==2:
                    # use t-statistic as a measure of discriminability
                    # larger pos value means resp[label==1] > resp[label==0]
                    sem_discrim_each_axis[vv,aa] = stats_utils.ttest_warn(groups[1], groups[0]).statistic
                else:
                    # if more than 2 classes, computing an F statistic 
                    sem_discrim_each_axis[vv,aa] = stats_utils.anova_oneway_warn(groups).statistic
                # also computing a correlation coefficient between semantic label/voxel response
                # sign is consistent with t-statistic
                sem_corr_each_axis[vv,aa] = stats_utils.numpy_corrcoef_warn(\
                                        resp[inds2use],labels[inds2use])[0,1]
                for gi, gg in enumerate(groups):
                    n_samp_each_axis[vv,aa,gi] = len(gg)
                    # mean within each label group 
                    mean_each_sem_level[vv,aa,gi] = np.mean(gg)
            else:                
                # at least one category is missing for this voxel's pRF and this semantic axis.
                # skip it and put nans in the arrays.               
                sem_discrim_each_axis[vv,aa] = np.nan
                sem_corr_each_axis[vv,aa] = np.nan
                n_samp_each_axis[vv,aa,:] = np.nan
                mean_each_sem_level[vv,aa,:] = np.nan
                
    return sem_discrim_each_axis, sem_corr_each_axis, n_samp_each_axis, mean_each_sem_level


def get_semantic_partial_corrs(best_prf_inds, labels_all, axes_to_do, \
                               unique_labels_each, val_voxel_data_pred, \
                               trials_use_each_prf = None, debug=False):   
    """
    Measure how well voxels' predicted responses distinguish between image patches with 
    different semantic content.
    Computing partial correlation coefficients here - to disentangle contributions of different 
    (possibly correlated) semantic features.
    """

    n_trials, n_voxels = val_voxel_data_pred.shape[0:2]

    n_sem_axes = len(axes_to_do)
    labels_all = labels_all[:,axes_to_do,:]
    
    partial_corr_each_axis = np.zeros((n_voxels, n_sem_axes))
    
    max_categ = np.max([len(unique_labels_each[aa]) for aa in axes_to_do])
    n_samp_each_axis = np.zeros((n_voxels, n_sem_axes, max_categ),dtype=np.float32)
   
    for vv in range(n_voxels):
        
        if debug and (vv>1):
            continue
        print('computing partial correlations for voxel %d of %d, prf %d\n'%(vv, n_voxels, best_prf_inds[vv,0]))
        
        resp = val_voxel_data_pred[:,vv,0]
        
        if trials_use_each_prf is not None:
            # select subset of trials to work with
            trials_use = trials_use_each_prf[:,best_prf_inds[vv,0]]
            resp = resp[trials_use]
            labels_use = labels_all[trials_use,:,:]
            if np.sum(trials_use)==0:
                print('voxel %d: no trials are included here, skipping it'%vv)
                partial_corr_each_axis[vv,:] = np.nan
                continue
        else:
            labels_use = labels_all
        
        inds2use = (np.sum(np.isnan(labels_use[:,:,best_prf_inds[vv,0]]), axis=1)==0)
        
        for aa in range(n_sem_axes):
            
            other_axes = ~np.isin(np.arange(n_sem_axes), aa)
            
            # going to compute information about the current axis of interest, while
            # partialling out the other axes. 
            labels_main_axis = labels_use[:,aa,best_prf_inds[vv,0]]
            labels_other_axes = labels_use[:,other_axes,best_prf_inds[vv,0]]

            unique_labels_actual = np.unique(labels_main_axis[inds2use])
            
            if np.all(np.isin(unique_labels_each[aa], unique_labels_actual)):
                
                partial_corr = stats_utils.compute_partial_corr(x=labels_main_axis[inds2use], \
                                                                y=resp[inds2use], \
                                                                c=labels_other_axes[inds2use,:])
                partial_corr_each_axis[vv,aa] = partial_corr
                
                for ui, uu in enumerate(unique_labels_actual):
                    n_samp_each_axis[vv,aa,ui] = np.sum(labels_main_axis[inds2use]==uu)
                    
            else:                
                # at least one category is missing for this voxel's pRF and this semantic axis.
                # skip it and put nans in the arrays.               
                partial_corr_each_axis[vv,aa] = np.nan
                n_samp_each_axis[vv,aa,:] = np.nan
               
    return partial_corr_each_axis, n_samp_each_axis


