import sys, os
import numpy as np
import time, h5py

from utils import default_paths, numpy_utils, nsd_utils
from model_fitting import initialize_fitting 
from feature_extraction import texture_feature_utils
from sklearn import decomposition
import argparse
import pandas as pd

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

"""
Code to perform PCA on features within a given feature space (texture or contour etc).
PCA is done separately within each pRF position, and the results for all pRFs are saved in a single file.
"""

        
def run_pca_gabor(subject, min_pct_var=95, max_pc_to_retain=96, debug=False, which_prf_grid=1, \
                          save_dtype=np.float32, compress=True):

    path_to_load = default_paths.gabor_texture_feat_path

    n_ori=12; n_sf = 8;
    features_file = os.path.join(path_to_load, \
                 'S%d_features_each_prf_%dori_%dsf_gabor_solo_nonlin_grid%d.h5py'%\
                 (subject, n_ori, n_sf, which_prf_grid))
    if not os.path.exists(features_file):
        raise RuntimeError('Looking at %s for precomputed features, not found.'%features_file)   
    path_to_save = os.path.join(path_to_load, 'PCA')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid = which_prf_grid)    
    n_prfs = models.shape[0]
    
    print('Loading pre-computed features from %s'%features_file)
    t = time.time()
    with h5py.File(features_file, 'r') as data_set:
        values = np.copy(data_set['/features'])
        data_set.close() 
    elapsed = time.time() - t
    print('Took %.5f seconds to load file'%elapsed)
    features_each_prf = values

    # training / validation data always split the same way - shared 1000 inds are validation.
    subject_df = nsd_utils.get_subj_df(subject)
    valinds = np.array(subject_df['shared1000'])
    trninds = np.array(subject_df['shared1000']==False)
    n_trials = len(trninds)
    
    
    print('Size of features array for this image set is:')
    print(features_each_prf.shape)

    scores_each_prf = np.zeros((n_trials, max_pc_to_retain, n_prfs), dtype=save_dtype)
    actual_max_ncomp=0
   
    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue

        print('Processing pRF %d of %d'%(prf_model_index, n_prfs))
        
        features_in_prf = features_each_prf[:,:,prf_model_index]
        
        print('Size of features array for this image set and prf is:')
        print(features_in_prf.shape)
        print('any nans in array')
        print(np.any(np.isnan(features_in_prf)))
        
        # finding pca solution for just training data
        _, wts, pre_mean, ev = do_pca(features_in_prf[trninds,:], max_pc_to_retain=max_pc_to_retain)

        # now projecting all the data incl. val into same subspace
        feat_submean = features_in_prf - np.tile(pre_mean[np.newaxis,:], [features_in_prf.shape[0],1])
        scores = feat_submean @ wts.T
        
        n_comp_needed = np.where(np.cumsum(ev)>min_pct_var)
        if np.size(n_comp_needed)>0:
            n_comp_needed = n_comp_needed[0][0]+1
        else:
            n_comp_needed = scores.shape[1]
        print('Retaining %d components to explain %d pct var'%(n_comp_needed, min_pct_var))
        actual_max_ncomp = np.max([n_comp_needed, actual_max_ncomp])
        
        scores_each_prf[:,0:n_comp_needed,prf_model_index] = scores[:,0:n_comp_needed]
        scores_each_prf[:,n_comp_needed:,prf_model_index] = np.nan

    # To save space, get rid of portion of array that ended up all nans
    if debug:
        actual_max_ncomp=np.max([2,actual_max_ncomp])
        assert(np.all((scores_each_prf[:,actual_max_ncomp:,:]==0) | np.isnan(scores_each_prf[:,actual_max_ncomp:,:])))
    else:
        assert(np.all(np.isnan(scores_each_prf[:,actual_max_ncomp:,:])))
    scores_each_prf = scores_each_prf[:,0:actual_max_ncomp,:]
    print('final size of array to save:')
    print(scores_each_prf.shape)
    
    fn2save = os.path.join(path_to_save, 'S%d_%dori_%dsf_nonlin_PCA_grid%d.h5py'%\
                           (subject, n_ori, n_sf, which_prf_grid))
    print('saving to %s'%fn2save)
    t = time.time()
    with h5py.File(fn2save, 'w') as data_set:
        if compress==True:
            dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype, compression='gzip')
        else:
            dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype)
        data_set['/features'][:,:,:] = scores_each_prf
        data_set.close() 
    elapsed = time.time() - t

    print('Took %.5f sec to write file'%elapsed)
      
        
def run_pca_sketch_tokens(subject, min_pct_var=95, max_pc_to_retain=150, debug=False, which_prf_grid=1, \
                          save_dtype=np.float32, compress=True):

    path_to_load = default_paths.sketch_token_feat_path

    features_file = os.path.join(path_to_load, 'S%d_features_each_prf_grid%d.h5py'%(subject, which_prf_grid))
    if not os.path.exists(features_file):
        raise RuntimeError('Looking at %s for precomputed features, not found.'%features_file)   
    path_to_save = os.path.join(path_to_load, 'PCA')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid = which_prf_grid)    
    n_prfs = models.shape[0]
    
    print('Loading pre-computed features from %s'%features_file)
    t = time.time()
    with h5py.File(features_file, 'r') as data_set:
        values = np.copy(data_set['/features'])
        data_set.close() 
    elapsed = time.time() - t
    print('Took %.5f seconds to load file'%elapsed)
    features_each_prf = values

    zgroup_labels = np.concatenate([np.zeros(shape=(1,150)), np.ones(shape=(1,1))], axis=1)

    # training / validation data always split the same way - shared 1000 inds are validation.
    subject_df = nsd_utils.get_subj_df(subject)
    valinds = np.array(subject_df['shared1000'])
    trninds = np.array(subject_df['shared1000']==False)
    n_trials = len(trninds)
    
    
    print('Size of features array for this image set is:')
    print(features_each_prf.shape)

    scores_each_prf = np.zeros((n_trials, max_pc_to_retain, n_prfs), dtype=save_dtype)
    actual_max_ncomp=0
   
    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue

        print('Processing pRF %d of %d'%(prf_model_index, n_prfs))
        
        features_in_prf = features_each_prf[:,:,prf_model_index]
        
        features_in_prf_z = np.zeros_like(features_in_prf)
        features_in_prf_z[trninds,:] = numpy_utils.zscore_in_groups(features_in_prf[trninds,:], zgroup_labels)
        features_in_prf_z[~trninds,:] = numpy_utils.zscore_in_groups(features_in_prf[~trninds,:], zgroup_labels)
           
        print('Size of features array for this image set and prf is:')
        print(features_in_prf_z.shape)
        print('any nans in array')
        print(np.any(np.isnan(features_in_prf_z)))
        
        # finding pca solution for just training data
        _, wts, pre_mean, ev = do_pca(features_in_prf_z[trninds,:], max_pc_to_retain=max_pc_to_retain)

        # now projecting all the data incl. val into same subspace
        feat_submean = features_in_prf_z - np.tile(pre_mean[np.newaxis,:], [features_in_prf_z.shape[0],1])
        scores = feat_submean @ wts.T
        
        n_comp_needed = np.where(np.cumsum(ev)>min_pct_var)
        if np.size(n_comp_needed)>0:
            n_comp_needed = n_comp_needed[0][0]+1
        else:
            n_comp_needed = scores.shape[1]
        print('Retaining %d components to explain %d pct var'%(n_comp_needed, min_pct_var))
        actual_max_ncomp = np.max([n_comp_needed, actual_max_ncomp])
        
        scores_each_prf[:,0:n_comp_needed,prf_model_index] = scores[:,0:n_comp_needed]
        scores_each_prf[:,n_comp_needed:,prf_model_index] = np.nan

    # To save space, get rid of portion of array that ended up all nans
    if debug:
        actual_max_ncomp=np.max([2,actual_max_ncomp])
        assert(np.all((scores_each_prf[:,actual_max_ncomp:,:]==0) | np.isnan(scores_each_prf[:,actual_max_ncomp:,:])))
    else:
        assert(np.all(np.isnan(scores_each_prf[:,actual_max_ncomp:,:])))
    scores_each_prf = scores_each_prf[:,0:actual_max_ncomp,:]
    print('final size of array to save:')
    print(scores_each_prf.shape)
    
    fn2save = os.path.join(path_to_save, 'S%d_PCA_grid%d.h5py'%(subject, which_prf_grid))
    print('saving to %s'%fn2save)
    t = time.time()
    with h5py.File(fn2save, 'w') as data_set:
        if compress==True:
            dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype, compression='gzip')
        else:
            dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype)
        data_set['/features'][:,:,:] = scores_each_prf
        data_set.close() 
    elapsed = time.time() - t

    print('Took %.5f sec to write file'%elapsed)

    
def run_pca_alexnet(subject, layer_name, min_pct_var=95, max_pc_to_retain=None, debug=False, \
                    which_prf_grid=1, save_dtype=np.float32, compress=True):

    path_to_load = default_paths.alexnet_feat_path

    features_file = os.path.join(path_to_load, 'S%d_%s_ReLU_reflect_features_each_prf_grid%d.h5py'%\
                                 (subject, layer_name, which_prf_grid))
    if not os.path.exists(features_file):
        raise RuntimeError('Looking at %s for precomputed features, not found.'%features_file)   
    with h5py.File(features_file, 'r') as data_set:
        dsize = data_set['/features'].shape
    n_features = dsize[1]
    n_trials = dsize[0]
    
    path_to_save = os.path.join(path_to_load, 'PCA')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid=which_prf_grid)    
    
    n_prfs = models.shape[0]
    assert(n_prfs==dsize[2])
    prf_batch_size = 50 # batching prfs for loading
    n_prf_batches = int(np.ceil(n_prfs/prf_batch_size))          
    prf_batch_inds = [np.arange(prf_batch_size*bb, np.min([prf_batch_size*(bb+1), n_prfs])) for bb in range(n_prf_batches)]
    prf_inds_loaded = []
    
    zgroup_labels = np.ones(shape=(1,n_features))

    # training / validation data always split the same way - shared 1000 inds are validation.
    subject_df = nsd_utils.get_subj_df(subject)
    valinds = np.array(subject_df['shared1000'])
    trninds = np.array(subject_df['shared1000']==False)

    scores_each_prf = np.zeros((n_trials, max_pc_to_retain, n_prfs), dtype=save_dtype)
    actual_max_ncomp=0
   
    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue

        print('Processing pRF %d of %d'%(prf_model_index, n_prfs))
        if prf_model_index not in prf_inds_loaded:

            batch_to_use = np.where([prf_model_index in prf_batch_inds[bb] for \
                                     bb in range(len(prf_batch_inds))])[0][0]
            assert(prf_model_index in prf_batch_inds[batch_to_use])
            print('Loading pre-computed features for prf models [%d - %d] from %s'%\
                  (prf_batch_inds[batch_to_use][0],prf_batch_inds[batch_to_use][-1], features_file))
            features_each_prf_batch = None
            prf_inds_loaded = prf_batch_inds[batch_to_use]
            
            t = time.time()
            with h5py.File(features_file, 'r') as data_set:
                values = np.copy(data_set['/features'][:,:,prf_batch_inds[batch_to_use]])
                data_set.close() 
            elapsed = time.time() - t
            print('Took %.5f seconds to load file'%elapsed)
            features_each_prf_batch = values
 
            print('Size of features array for this image set and prf batch is:')
            print(features_each_prf_batch.shape)

        index_into_batch = np.where(prf_model_index==prf_inds_loaded)[0][0]
        print('Index into batch for prf %d: %d'%(prf_model_index, index_into_batch))
        features_in_prf = features_each_prf_batch[:,:,index_into_batch]
        
        features_in_prf_z = np.zeros_like(features_in_prf)
        features_in_prf_z[trninds,:] = numpy_utils.zscore_in_groups(features_in_prf[trninds,:], zgroup_labels)
        features_in_prf_z[~trninds,:] = numpy_utils.zscore_in_groups(features_in_prf[~trninds,:], zgroup_labels)
        # if any feature channels had no variance, fix them now
        zero_var = (np.var(features_in_prf[trninds,:], axis=0)==0) | \
                    (np.var(features_in_prf[~trninds,:], axis=0)==0)
        features_in_prf_z[:,zero_var] = features_in_prf_z[0,zero_var]
        
        print('Size of features array for this image set and prf is:')
        print(features_in_prf_z.shape)
        print('any nans in array')
        print(np.any(np.isnan(features_in_prf_z)))
        
        # finding pca solution for just training data
        _, wts, pre_mean, ev = do_pca(features_in_prf_z[trninds,:], max_pc_to_retain=max_pc_to_retain)

        # now projecting all the data incl. val into same subspace
        feat_submean = features_in_prf_z - np.tile(pre_mean[np.newaxis,:], [features_in_prf_z.shape[0],1])
        scores = feat_submean @ wts.T
        
        n_comp_needed = np.where(np.cumsum(ev)>min_pct_var)
        if np.size(n_comp_needed)>0:
            n_comp_needed = n_comp_needed[0][0]+1
        else:
            n_comp_needed = scores.shape[1]
        print('Retaining %d components to explain %d pct var'%(n_comp_needed, min_pct_var))
        actual_max_ncomp = np.max([n_comp_needed, actual_max_ncomp])
        
        scores_each_prf[:,0:n_comp_needed,prf_model_index] = scores[:,0:n_comp_needed]
        scores_each_prf[:,n_comp_needed:,prf_model_index] = np.nan

    # To save space, get rid of portion of array that ended up all nans
    if debug:
        actual_max_ncomp=np.max([2,actual_max_ncomp])
        assert(np.all((scores_each_prf[:,actual_max_ncomp:,:]==0) | np.isnan(scores_each_prf[:,actual_max_ncomp:,:])))
    else:
        assert(np.all(np.isnan(scores_each_prf[:,actual_max_ncomp:,:])))
    scores_each_prf = scores_each_prf[:,0:actual_max_ncomp,:]
    print('final size of array to save:')
    print(scores_each_prf.shape)
    
    fn2save = os.path.join(path_to_save, 'S%d_%s_ReLU_reflect_PCA_grid%d.h5py'%(subject, layer_name, which_prf_grid))
    print('saving to %s'%fn2save)
    
    t = time.time()
    with h5py.File(fn2save, 'w') as data_set:
        if compress==True:
            dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype, compression='gzip')
        else:
            dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype)
        data_set['/features'][:,:,:] = scores_each_prf
        data_set.close() 
    elapsed = time.time() - t

    print('Took %.5f sec to write file'%elapsed)


def run_pca_clip(subject, layer_name, min_pct_var=95, max_pc_to_retain=None, debug=False, \
                     which_prf_grid=1, save_dtype=np.float32, compress=True):

    path_to_load = default_paths.clip_feat_path

    model_architecture = 'RN50'
    features_file = os.path.join(path_to_load, 'S%d_%s_%s_features_each_prf_grid%d_prfbatch%d.h5py'%\
                                 (subject, model_architecture,layer_name, which_prf_grid, 0))
    if not os.path.exists(features_file):
        raise RuntimeError('Looking at %s for precomputed features, not found.'%features_file)   
    with h5py.File(features_file, 'r') as data_set:
        dsize = data_set['/features'].shape
    n_features = dsize[1]
    n_trials = dsize[0]

    path_to_save = os.path.join(path_to_load, 'PCA')
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid=which_prf_grid)    
    
    n_prfs = models.shape[0] 
    prf_batch_size = 100 # batching prfs for loading
    assert(prf_batch_size==dsize[2])
    n_prf_batches = int(np.ceil(n_prfs/prf_batch_size))          
    prf_batch_inds = [np.arange(prf_batch_size*bb, np.min([prf_batch_size*(bb+1), n_prfs])) for bb in range(n_prf_batches)]
    prf_inds_loaded = []
        
    zgroup_labels = np.ones(shape=(1,n_features))

    # training / validation data always split the same way - shared 1000 inds are validation.
    subject_df = nsd_utils.get_subj_df(subject)
    valinds = np.array(subject_df['shared1000'])
    trninds = np.array(subject_df['shared1000']==False)

    scores_each_prf = np.zeros((n_trials, max_pc_to_retain, n_prfs), dtype=save_dtype)
    actual_max_ncomp=0
    
    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue

        print('Processing pRF %d of %d'%(prf_model_index, n_prfs))
        if prf_model_index not in prf_inds_loaded:

            batch_to_use = np.where([prf_model_index in prf_batch_inds[bb] for \
                                     bb in range(len(prf_batch_inds))])[0][0]
            assert(prf_model_index in prf_batch_inds[batch_to_use])
            # each batch of prfs is in a separate file here
            features_file = os.path.join(path_to_load, 'S%d_%s_%s_features_each_prf_grid%d_prfbatch%d.h5py'%\
                                 (subject, model_architecture,layer_name, which_prf_grid, batch_to_use))
            print('Loading pre-computed features for prf models [%d - %d] from %s'%\
                  (prf_batch_inds[batch_to_use][0],prf_batch_inds[batch_to_use][-1], features_file))
            features_each_prf_batch = None
            prf_inds_loaded = prf_batch_inds[batch_to_use]
            
            t = time.time()
            
            with h5py.File(features_file, 'r') as data_set:
                values = np.copy(data_set['/features'][:,:,:])
                data_set.close() 
            elapsed = time.time() - t
            print('Took %.5f seconds to load file'%elapsed)
            features_each_prf_batch = values
 
            print('Size of features array for this image set and prf batch is:')
            print(features_each_prf_batch.shape)

        index_into_batch = np.where(prf_model_index==prf_inds_loaded)[0][0]
        print('Index into batch for prf %d: %d'%(prf_model_index, index_into_batch))
        features_in_prf = features_each_prf_batch[:,:,index_into_batch]
        
        features_in_prf_z = np.zeros_like(features_in_prf)
        features_in_prf_z[trninds,:] = numpy_utils.zscore_in_groups(features_in_prf[trninds,:], zgroup_labels)
        features_in_prf_z[~trninds,:] = numpy_utils.zscore_in_groups(features_in_prf[~trninds,:], zgroup_labels)
        # if any feature channels had no variance, fix them now
        zero_var = (np.var(features_in_prf[trninds,:], axis=0)==0) | \
                    (np.var(features_in_prf[~trninds,:], axis=0)==0)
        features_in_prf_z[:,zero_var] = features_in_prf_z[0,zero_var]
        
        print('Size of features array for this image set and prf is:')
        print(features_in_prf_z.shape)

        print('any nans in array')
        print(np.any(np.isnan(features_in_prf_z)))
        # finding pca solution for just training data
        _, wts, pre_mean, ev = do_pca(features_in_prf_z[trninds,:], max_pc_to_retain=max_pc_to_retain)

        # now projecting all the data incl. val into same subspace
        feat_submean = features_in_prf_z - np.tile(pre_mean[np.newaxis,:], [features_in_prf_z.shape[0],1])
        scores = feat_submean @ wts.T
        
        n_comp_needed = np.where(np.cumsum(ev)>min_pct_var)
        if np.size(n_comp_needed)>0:
            n_comp_needed = n_comp_needed[0][0]+1
        else:
            n_comp_needed = scores.shape[1]
        print('Retaining %d components to explain %d pct var'%(n_comp_needed, min_pct_var))
        actual_max_ncomp = np.max([n_comp_needed, actual_max_ncomp])
        
        scores_each_prf[:,0:n_comp_needed,prf_model_index] = scores[:,0:n_comp_needed]
        scores_each_prf[:,n_comp_needed:,prf_model_index] = np.nan
     
    # To save space, get rid of portion of array that ended up all nans
    if debug:
        actual_max_ncomp=np.max([2,actual_max_ncomp])
        assert(np.all((scores_each_prf[:,actual_max_ncomp:,:]==0) | np.isnan(scores_each_prf[:,actual_max_ncomp:,:])))
    else:
        assert(np.all(np.isnan(scores_each_prf[:,actual_max_ncomp:,:])))
    scores_each_prf = scores_each_prf[:,0:actual_max_ncomp,:]
    print('final size of array to save:')
    print(scores_each_prf.shape)
    
    fn2save = os.path.join(path_to_save, 'S%d_%s_%s_PCA_grid%d.h5py'%(subject, model_architecture, \
                                                                              layer_name, which_prf_grid))
    print('saving to %s'%fn2save)
    
    t = time.time()
    with h5py.File(fn2save, 'w') as data_set:
        if compress==True:
            dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype, compression='gzip')
        else:
            dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype)
        data_set['/features'][:,:,:] = scores_each_prf
        data_set.close() 
    elapsed = time.time() - t

    print('Took %.5f sec to write file'%elapsed)

    
def run_pca_semantic(subject, semantic_feature_set, min_pct_var=95, max_pc_to_retain=92, debug=False,\
                     which_prf_grid=1, save_dtype=np.float32):

    labels_folder = os.path.join(default_paths.stim_labels_root, \
                                         'S%d_within_prf_grid%d'%(subject, which_prf_grid))
    path_to_save = os.path.join(default_paths.stim_labels_root, \
                                         'S%d_within_prf_grid%d_PCA'%(subject, which_prf_grid))
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
        
    if semantic_feature_set=='coco_things_categ':
        features_file = os.path.join(labels_folder, \
                                  'S%d_cocolabs_binary_prf0.csv'%(subject))
        n_features_orig = 80
    elif semantic_feature_set=='coco_stuff_categ':
        features_file = os.path.join(labels_folder, \
                                  'S%d_cocolabs_stuff_binary_prf0.csv'%(subject)) 
        n_features_orig = 92
    else:
        raise ValueError('feature set %s not recognized'%semantic_feature_set)
    
    if not os.path.exists(features_file):
        raise RuntimeError('Looking at %s for raw semantic features, not found.'%features_file)
      
    
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid = which_prf_grid)    
    n_prfs = models.shape[0]
   
    # training / validation data always split the same way - shared 1000 inds are validation.
    subject_df = nsd_utils.get_subj_df(subject)
    valinds = np.array(subject_df['shared1000'])
    trninds = np.array(subject_df['shared1000']==False)
    n_trials = len(trninds)
    
    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue

        print('Processing pRF %d of %d'%(prf_model_index, n_prfs))
        
        fn2save = os.path.join(path_to_save, 'S%d_%s_prf%d_PCA.csv'%(subject, semantic_feature_set, prf_model_index))

        if 'stuff' in semantic_feature_set:
            features_file = os.path.join(labels_folder, \
                                  'S%d_cocolabs_stuff_binary_prf%d.csv'%(subject, prf_model_index))
            print('Loading pre-computed features from %s'%features_file)
            coco_df = pd.read_csv(features_file, index_col=0)
            labels = np.array(coco_df)[:,16:108]  
        else:
            features_file = os.path.join(labels_folder, \
                                  'S%d_cocolabs_binary_prf%d.csv'%(subject, prf_model_index))
            print('Loading pre-computed features from %s'%features_file)
            coco_df = pd.read_csv(features_file, index_col=0)
            labels = np.array(coco_df)[:,12:92]   
           
        features_in_prf = labels.astype(np.float32)
        
        assert(features_in_prf.shape[0]==n_trials)
        assert(features_in_prf.shape[1]==n_features_orig)
          
        print('Size of features array for this image set and prf is:')
        print(features_in_prf.shape)
        print('any nans in array')
        print(np.any(np.isnan(features_in_prf)))
        print('num each column:')
        print(np.sum(features_in_prf, axis=0).astype(int))
        
        # finding pca solution for just training data
        _, wts, pre_mean, ev = do_pca(features_in_prf[trninds,:], max_pc_to_retain=max_pc_to_retain)

        # now projecting all the data incl. val into same subspace
        feat_submean = features_in_prf - np.tile(pre_mean[np.newaxis,:], [features_in_prf.shape[0],1])
        scores = feat_submean @ wts.T
        
        n_comp_needed = np.where(np.cumsum(ev)>min_pct_var)
        if np.size(n_comp_needed)>0:
            n_comp_needed = n_comp_needed[0][0]+1
        else:
            n_comp_needed = scores.shape[1]
        print('Retaining %d components to explain %d pct var'%(n_comp_needed, min_pct_var))
       
        # save as a csv file for easy loading later on
        scores_df = pd.DataFrame(data=scores[:,0:n_comp_needed],dtype=save_dtype)
        
        print('Saving to %s'%fn2save)
        scores_df.to_csv(fn2save, header=True)
        


def do_pca(values, max_pc_to_retain=None):
    """
    Apply PCA to the data, return reduced dim data as well as weights, var explained.
    """
    n_features_actual = values.shape[1]
    n_trials = values.shape[0]
    
    if max_pc_to_retain is not None:        
        n_comp = np.min([np.min([max_pc_to_retain, n_features_actual]), n_trials])
    else:
        n_comp = np.min([n_features_actual, n_trials])
         
    print('Running PCA: original size of array is [%d x %d]'%(n_trials, n_features_actual))
    t = time.time()
    pca = decomposition.PCA(n_components = n_comp, copy=False)
    scores = pca.fit_transform(values)           
    elapsed = time.time() - t
    print('Time elapsed: %.5f'%elapsed)
    values = None            
    wts = pca.components_
    ev = pca.explained_variance_
    ev = ev/np.sum(ev)*100
    pre_mean = pca.mean_
  
    return scores, wts, pre_mean, ev


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--type", type=str,default='sketch_tokens',
                    help="what kind of features are we using?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--zscore", type=int,default=0,
                    help="want to zscore individual columns before pca? 1 for yes, 0 for no")
    parser.add_argument("--max_pc_to_retain", type=int,default=0,
                    help="max pc to retain? enter 0 for None")
    parser.add_argument("--min_pct_var", type=int,default=95,
                    help="min pct var to explain? default 95")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which pRF grid to use?")
    
    args = parser.parse_args()
    
    print('\nusing prf grid %d\n'%(args.which_prf_grid))
    
    if args.max_pc_to_retain==0:
        args.max_pc_to_retain = None
    
    if args.type=='sketch_tokens':
        run_pca_sketch_tokens(subject=args.subject, min_pct_var=args.min_pct_var, \
                              max_pc_to_retain=args.max_pc_to_retain, debug=args.debug==1, \
                              which_prf_grid=args.which_prf_grid)
    elif args.type=='gabor':
        run_pca_gabor(subject=args.subject, min_pct_var=args.min_pct_var, \
                              max_pc_to_retain=args.max_pc_to_retain, debug=args.debug==1, \
                              which_prf_grid=args.which_prf_grid)
    
    elif args.type=='alexnet':
        layers = ['Conv%d'%(ll+1) for ll in range(5)]
        for layer in layers:
            run_pca_alexnet(subject=args.subject, layer_name=layer, min_pct_var=args.min_pct_var, \
                            max_pc_to_retain=args.max_pc_to_retain, debug=args.debug==1, \
                            which_prf_grid=args.which_prf_grid)
    elif args.type=='clip':
        layers = ['block%d'%(ll) for ll in range(16)]
        for layer in layers:
            run_pca_clip(subject=args.subject, layer_name=layer, min_pct_var=args.min_pct_var, \
                         max_pc_to_retain=args.max_pc_to_retain, debug=args.debug==1, \
                         which_prf_grid=args.which_prf_grid)    
    elif args.type=='semantic_stuff':
        run_pca_semantic(subject=args.subject, semantic_feature_set='coco_stuff_categ', \
                         min_pct_var=args.min_pct_var, \
                         max_pc_to_retain=args.max_pc_to_retain, debug=args.debug==1, \
                         which_prf_grid=args.which_prf_grid)
    elif args.type=='semantic_things':
        run_pca_semantic(subject=args.subject, semantic_feature_set='coco_things_categ', \
                         min_pct_var=args.min_pct_var, \
                         max_pc_to_retain=args.max_pc_to_retain, debug=args.debug==1, \
                         which_prf_grid=args.which_prf_grid)
    else:
        raise ValueError('--type %s is not recognized'%args.type)