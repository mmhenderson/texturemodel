import numpy as np
from utils import stats_utils

def get_feature_tuning(best_prf_inds, features_each_prf, val_voxel_data_pred, \
                       trials_use_each_prf = None, debug=False):
   
    """
    Get an approximate measure of voxels' tuning for particular features, based on how correlated 
    the predicted responses of the encoding model are with the activation in each feature channel.
    """
    
    n_trials = val_voxel_data_pred.shape[0]
    n_voxels = val_voxel_data_pred.shape[1]
    n_features = features_each_prf.shape[1]
    corr_each_feature = np.zeros((n_voxels, n_features))
   
    for vv in range(n_voxels):
        
        if debug and (vv>1):
            continue
        print('computing feature tuning for voxel %d of %d, prf %d\n'%(vv, n_voxels, best_prf_inds[vv,0]))
           
        # voxel's predicted response to validation set trials, based on encoding model.
        resp = val_voxel_data_pred[:,vv,0]
        # activation in the feature of interest on each trial.
        feat_act = features_each_prf[:,:,best_prf_inds[vv,0]]
        
        if trials_use_each_prf is not None:
            # select subset of trials to work with
            trials_use = trials_use_each_prf[:,best_prf_inds[vv,0]]
            resp = resp[trials_use]
            feat_act = feat_act[trials_use]
            if np.sum(trials_use)==0:
                print('voxel %d: no trials are included here, skipping it'%vv)
                corr_each_feature[vv,:] = np.nan
                continue
        
        for ff in range(n_features):
            if np.var(feat_act[:,ff])>0:
                corr_each_feature[vv,ff] = stats_utils.numpy_corrcoef_warn(resp, feat_act[:,ff])[0,1]
            else:
                corr_each_feature[vv,ff] = np.nan
                
    return corr_each_feature

