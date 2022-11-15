import sys, os
import numpy as np
import time, h5py

from utils import default_paths, numpy_utils, nsd_utils
from model_fitting import initialize_fitting 
from sklearn import decomposition
import argparse
import pandas as pd

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

"""
Code to perform PCA on features within a given feature space (texture or contour etc).
PCA is done separately within each pRF position, and the results for all pRFs are saved in a single file.
"""

def run_pca_each_prf(raw_filename, pca_filename, \
            fit_inds=None, prf_batch_size=50, \
              zscore_before_pca = False, \
              zgroup_labels = None,\
              min_pct_var=95, max_pc_to_retain=100,\
              save_weights=False, save_weights_filename=None,\
              use_saved_ncomp=False, ncomp_filename=None,\
              save_dtype=np.float32, compress=True, \
              debug=False):

    if isinstance(raw_filename, list):  
        n_prf_batches = len(raw_filename) 
        batches_in_separate_files=True        
    else:
        raw_filename = [raw_filename]
        
        batches_in_separate_files=False
        
    
    n_prfs=0;
    n_feat_orig = 0;
    for fi, fn in enumerate(raw_filename):
        print(fn)
        if not os.path.exists(fn):
            raise RuntimeError('Looking at %s for precomputed features, not found.'%fn)   
        with h5py.File(fn, 'r') as data_set:
            dsize = data_set['/features'].shape
            data_set.close() 

        n_trials, nf, n_prfs_tmp = dsize
        print('nf=%d'%nf)
        
        n_feat_orig = np.maximum(n_feat_orig, nf)
        
        if fi==0 and batches_in_separate_files:
            assert(n_prfs_tmp==prf_batch_size)
            nf_1 = nf
        elif fi>0 and batches_in_separate_files:
            # did the features from different prfs have diff sizes, before pca?
            prfs_diff_orig_sizes = (nf_1!=nf)
        else:
            prfs_diff_orig_sizes = False 
            
        n_prfs += n_prfs_tmp
    
    print('prfs_diff_orig_sizes=%s'%prfs_diff_orig_sizes)
    
    if not batches_in_separate_files:
        n_prf_batches = int(np.ceil(n_prfs/prf_batch_size))  
    
    print('max_pc_to_retain:')
    print(max_pc_to_retain, n_feat_orig)
    
    if max_pc_to_retain is None:
        max_pc_to_retain = n_feat_orig
    else:
        max_pc_to_retain = np.minimum(max_pc_to_retain, n_feat_orig)
    print(max_pc_to_retain)
    
    if fit_inds is not None:
        assert(len(fit_inds)==n_trials)
    else:
        fit_inds = np.ones((n_trials,),dtype=bool)
        
    # batching prfs for loading    
    prf_batch_inds = [np.arange(prf_batch_size*bb, np.min([prf_batch_size*(bb+1), n_prfs])) \
                          for bb in range(n_prf_batches)]
    prf_inds_loaded = []
    
    if zscore_before_pca and zgroup_labels is None:
        zgroup_labels = np.ones(shape=(1,n_feat_orig))
        
    scores_each_prf = np.zeros((n_trials, max_pc_to_retain, n_prfs), dtype=save_dtype)
    actual_max_ncomp=0
    
    if save_weights:
        
        if prfs_diff_orig_sizes:
            pca_wts = [[] for mm in range(n_prfs)]
            pca_premean = [[] for mm in range(n_prfs)]
        else:
            pca_wts = np.zeros((n_feat_orig, max_pc_to_retain, n_prfs), dtype=save_dtype)
            pca_premean = np.zeros((n_feat_orig, n_prfs), dtype=save_dtype)
        pca_ncomp = np.zeros((n_prfs,),dtype=save_dtype)
        assert(save_weights_filename is not None)
        
    if use_saved_ncomp:
        print('loading ncomp from %s'%ncomp_filename)
        saved_pca_ncomp = np.load(ncomp_filename).astype(int)
        
    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue

        print('Processing pRF %d of %d'%(prf_model_index, n_prfs))

        if prf_model_index not in prf_inds_loaded:

            batch_to_use = np.where([prf_model_index in prf_batch_inds[bb] for \
                                     bb in range(len(prf_batch_inds))])[0][0]
            features_each_prf_batch = None
            prf_inds_loaded = prf_batch_inds[batch_to_use]
            
            t = time.time()
            
            if batches_in_separate_files:
                
                # each batch of prfs is in a separate file here
                filename_load = raw_filename[batch_to_use]
                print('Loading pre-computed features for prf models [%d - %d] from %s'%\
                  (prf_batch_inds[batch_to_use][0],prf_batch_inds[batch_to_use][-1], filename_load))
                with h5py.File(filename_load, 'r') as data_set:
                    values = np.copy(data_set['/features'][:,:,:])
                    data_set.close() 
                
            else:
                
                # all prfs are in same file, can just load part of it
                filename_load = raw_filename[0]
                print('Loading pre-computed features for prf models [%d - %d] from %s'%\
                    (prf_batch_inds[batch_to_use][0],prf_batch_inds[batch_to_use][-1], filename_load))
                with h5py.File(filename_load, 'r') as data_set:
                    values = np.copy(data_set['/features'][:,:,prf_batch_inds[batch_to_use]])
                    data_set.close() 
                
            features_each_prf_batch = values

        index_into_batch = np.where(prf_model_index==prf_inds_loaded)[0][0]
        print('Index into batch for prf %d: %d'%(prf_model_index, index_into_batch))
        features_in_prf = features_each_prf_batch[:,:,index_into_batch]
        n_feat_orig_actual = features_in_prf.shape[1]
        
        elapsed = time.time() - t
        print('Took %.5f seconds to load file'%elapsed)
        
        if zscore_before_pca:
            features_in_prf_z = np.zeros_like(features_in_prf)
            features_in_prf_z[fit_inds,:] = numpy_utils.zscore_in_groups(features_in_prf[fit_inds,:], \
                                                                         zgroup_labels[0:n_feat_orig_actual])
            zero_var = np.var(features_in_prf[fit_inds,:], axis=0)==0
            if np.sum(~fit_inds)>0:
                features_in_prf_z[~fit_inds,:] = numpy_utils.zscore_in_groups(features_in_prf[~fit_inds,:], \
                                                                              zgroup_labels[0:n_feat_orig_actual])
                # if any feature channels had no variance, fix them now
                zero_var = zero_var | (np.var(features_in_prf[~fit_inds,:], axis=0)==0)
            print('there are %d columns with zero variance'%np.sum(zero_var))
            features_in_prf_z[:,zero_var] = features_in_prf_z[0,zero_var]
       
            features_in_prf = features_in_prf_z
        
        print('Size of features array for this image set and prf is:')
        print(features_in_prf.shape)
        print('any nans in array: %s'%np.any(np.isnan(features_in_prf)))
        
        # finding pca solution for just training data (specified in fit_inds)
        _, wts, pre_mean, ev = compute_pca(features_in_prf[fit_inds,:], max_pc_to_retain=max_pc_to_retain)

        # now projecting all the data incl. val into same subspace
        feat_submean = features_in_prf - np.tile(pre_mean[np.newaxis,:], [features_in_prf.shape[0],1])
        scores = feat_submean @ wts.T
        
        if use_saved_ncomp:
            n_comp_needed = saved_pca_ncomp[prf_model_index]
        else:
            n_comp_needed = np.where(np.cumsum(ev)>min_pct_var)
            if np.size(n_comp_needed)>0:
                n_comp_needed = n_comp_needed[0][0]+1
            else:
                n_comp_needed = scores.shape[1]
                
        print('Retaining %d components to explain %d pct var'%(n_comp_needed, min_pct_var))
        actual_max_ncomp = np.max([n_comp_needed, actual_max_ncomp])
        
        scores_each_prf[:,0:n_comp_needed,prf_model_index] = scores[:,0:n_comp_needed]
        scores_each_prf[:,n_comp_needed:,prf_model_index] = np.nan

        if save_weights:
            if prfs_diff_orig_sizes:
                pca_wts[prf_model_index] = wts.T
                pca_premean[prf_model_index] = pre_mean
            else:
                pca_wts[:,:,prf_model_index] = wts.T
                pca_premean[:,prf_model_index] = pre_mean
            pca_ncomp[prf_model_index] = n_comp_needed
            
    # To save space, get rid of portion of array that ended up all nans
    if debug:
        actual_max_ncomp=np.max([2,actual_max_ncomp])
        assert(np.all((scores_each_prf[:,actual_max_ncomp:,:]==0) | np.isnan(scores_each_prf[:,actual_max_ncomp:,:])))
    else:
        assert(np.all(np.isnan(scores_each_prf[:,actual_max_ncomp:,:])))
    scores_each_prf = scores_each_prf[:,0:actual_max_ncomp,:]
    
    print('final size of array to save:')
    print(scores_each_prf.shape)    
    print('saving to %s'%pca_filename)
    
    t = time.time()
    
    with h5py.File(pca_filename, 'w') as data_set:
        if compress==True:
            dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype, compression='gzip')
        else:
            dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype)
        data_set['/features'][:,:,:] = scores_each_prf
        data_set.close() 
    elapsed = time.time() - t

    print('Took %.5f sec to write file'%elapsed)
      
    if save_weights:
        wts = {'pca_wts': pca_wts, 'pca_premean': pca_premean, 'pca_ncomp': pca_ncomp}
        print('saving the weights for this pca to %s'%save_weights_filename)
        np.save(save_weights_filename, wts, allow_pickle=True)

        
def apply_pca_each_prf(raw_filename, pca_filename, \
                       load_weights_filename, \
              prf_batch_size=50, \
              zscore_before_pca = False, \
              zgroup_labels = None,\
              save_dtype=np.float32, compress=True, \
              debug=False):

    if isinstance(raw_filename, list):  
        n_prf_batches = len(raw_filename) 
        batches_in_separate_files=True        
    else:
        raw_filename = [raw_filename]      
        batches_in_separate_files=False

    print('loading pre-computed pca weights from %s'%(load_weights_filename))
    w = np.load(load_weights_filename, allow_pickle=True).item()
    pca_wts = w['pca_wts']
    pca_premean = w['pca_premean']
    pca_ncomp = w['pca_ncomp']

    n_prfs=0;
    n_feat_orig = 0;
    for fi, fn in enumerate(raw_filename):
        if not os.path.exists(fn):
            raise RuntimeError('Looking at %s for precomputed features, not found.'%fn)   
        with h5py.File(fn, 'r') as data_set:
            dsize = data_set['/features'].shape
            data_set.close() 

        n_trials, nf, n_prfs_tmp = dsize
        print('nf=%d'%nf)
        
        n_feat_orig = np.maximum(n_feat_orig, nf)
        
        if fi==0 and batches_in_separate_files:
            assert(n_prfs_tmp==prf_batch_size)
            nf_1 = nf
        elif fi>0 and batches_in_separate_files:
            # did the features from different prfs have diff sizes, before pca?
            prfs_diff_orig_sizes = (nf_1!=nf)
        else:
            prfs_diff_orig_sizes = False
            
        n_prfs += n_prfs_tmp
    
    print('prfs_diff_orig_sizes=%s'%prfs_diff_orig_sizes)
    if prfs_diff_orig_sizes:
        assert(not hasattr(pca_wts, 'shape') or len(pca_wts.shape)==1)
    else:
        assert(len(pca_wts.shape)==3)
        
    if not batches_in_separate_files:
        n_prf_batches = int(np.ceil(n_prfs/prf_batch_size))  
    
    max_pc_to_retain = n_feat_orig
    # batching prfs for loading    
    prf_batch_inds = [np.arange(prf_batch_size*bb, np.min([prf_batch_size*(bb+1), n_prfs])) \
                          for bb in range(n_prf_batches)]
    prf_inds_loaded = []
    
    if zscore_before_pca and zgroup_labels is None:
        zgroup_labels = np.ones(shape=(1,n_feat_orig))
        
    scores_each_prf = np.zeros((n_trials, max_pc_to_retain, n_prfs), dtype=save_dtype)
    actual_max_ncomp=0
    
    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue

        print('Processing pRF %d of %d'%(prf_model_index, n_prfs))

        if prf_model_index not in prf_inds_loaded:

            batch_to_use = np.where([prf_model_index in prf_batch_inds[bb] for \
                                     bb in range(len(prf_batch_inds))])[0][0]
            features_each_prf_batch = None
            prf_inds_loaded = prf_batch_inds[batch_to_use]
            
            t = time.time()
            
            if batches_in_separate_files:
                
                # each batch of prfs is in a separate file here
                filename_load = raw_filename[batch_to_use]
                print('Loading pre-computed features for prf models [%d - %d] from %s'%\
                  (prf_batch_inds[batch_to_use][0],prf_batch_inds[batch_to_use][-1], filename_load))
                with h5py.File(filename_load, 'r') as data_set:
                    values = np.copy(data_set['/features'][:,:,:])
                    data_set.close() 
                
            else:
                
                # all prfs are in same file, can just load part of it
                filename_load = raw_filename[0]
                print('Loading pre-computed features for prf models [%d - %d] from %s'%\
                    (prf_batch_inds[batch_to_use][0],prf_batch_inds[batch_to_use][-1], filename_load))
                with h5py.File(filename_load, 'r') as data_set:
                    values = np.copy(data_set['/features'][:,:,prf_batch_inds[batch_to_use]])
                    data_set.close() 
                
            features_each_prf_batch = values

        index_into_batch = np.where(prf_model_index==prf_inds_loaded)[0][0]
        print('Index into batch for prf %d: %d'%(prf_model_index, index_into_batch))
        features_in_prf = features_each_prf_batch[:,:,index_into_batch]
        n_feat_orig_actual = features_in_prf.shape[1]
        
        elapsed = time.time() - t
        print('Took %.5f seconds to load file'%elapsed)
        
        if zscore_before_pca:
            features_in_prf_z = numpy_utils.zscore_in_groups(features_in_prf, zgroup_labels[0:n_feat_orig_actual])
            zero_var = np.var(features_in_prf, axis=0)==0
            features_in_prf_z[:,zero_var] = features_in_prf_z[0,zero_var]
       
            features_in_prf = features_in_prf_z
        
        print('Size of features array for this image set and prf is:')
        print(features_in_prf.shape)
        print('any nans in array: %s'%np.any(np.isnan(features_in_prf)))
       
        if prfs_diff_orig_sizes:
            wts = pca_wts[prf_model_index]
            pre_mean = pca_pre_mean[prf_model_index]
        else:
            wts = pca_wts[:,:,prf_model_index]
            pre_mean = pca_premean[:,prf_model_index]
        n_comp_needed = int(pca_ncomp[prf_model_index])
        
        # project into pca subspace using saved wts
        feat_submean = features_in_prf - np.tile(pre_mean[np.newaxis,:], [features_in_prf.shape[0],1])
        scores = feat_submean @ wts
        
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
    print('saving to %s'%pca_filename)
    
    t = time.time()
    
    with h5py.File(pca_filename, 'w') as data_set:
        if compress==True:
            dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype, compression='gzip')
        else:
            dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype)
        data_set['/features'][:,:,:] = scores_each_prf
        data_set.close() 
    elapsed = time.time() - t

    print('Took %.5f sec to write file'%elapsed)

    
def run_pca_oneprf(features_raw, filename_save_pca, 
                    prf_save_ind=0, n_prfs_total=1, \
                    fit_inds=None, 
                    min_pct_var=95, max_pc_to_retain=100, 
                    save_weights=False, save_weights_filename=None, 
                    use_saved_ncomp=False, ncomp_filename=None,\
                    save_dtype=np.float32, compress=True, \
                    debug=False):

    n_trials, n_feat_orig = features_raw.shape
    
    print('max_pc_to_retain:')
    print(max_pc_to_retain)
    print('n_feat_orig:')
    print(n_feat_orig)
    
    if fit_inds is not None:
        assert(len(fit_inds)==n_trials)
    else:
        fit_inds = np.ones((n_trials,),dtype=bool)
      
    print('computing pca')
    sys.stdout.flush()
    # finding pca solution for just training data (specified in fit_inds)
    _, wts, pre_mean, ev = compute_pca(features_raw[fit_inds,:], max_pc_to_retain=max_pc_to_retain)

    # now projecting all the data incl. val into same subspace
    feat_submean = features_raw - np.tile(pre_mean[np.newaxis,:], [features_raw.shape[0],1])
    scores = feat_submean @ wts.T

    if use_saved_ncomp:
        print('loading ncomp from %s'%ncomp_filename)
        saved_pca_ncomp = np.load(ncomp_filename).astype(int)
        n_comp_needed = saved_pca_ncomp[0]
    else:
        n_comp_needed = np.where(np.cumsum(ev)>min_pct_var)
        if np.size(n_comp_needed)>0:
            n_comp_needed = n_comp_needed[0][0]+1
        else:
            n_comp_needed = scores.shape[1]

    print('Retaining %d components to explain %d pct var'%(n_comp_needed, min_pct_var))
    
    if n_prfs_total==1:
        scores = scores[:,0:n_comp_needed]
        nf = n_comp_needed
    else:
        scores[:, n_comp_needed:] = np.nan
        if scores.shape[1]<max_pc_to_retain:
            n_pad = max_pc_to_retain - scores.shape[1]
            print('padding array with nans for %d columns'%(n_pad))
            scores = np.concatenate([scores, np.full(shape=(n_trials, n_pad), \
                                                     fill_value=np.nan, dtype=scores.dtype)], axis=1)
        nf = max_pc_to_retain
        
    if save_weights:
        w = {'pca_wts': wts, 'pca_premean': pre_mean, 'pca_ncomp': np.array([n_comp_needed])}
        print('saving the weights for this pca to %s'%save_weights_filename)
        np.save(save_weights_filename, w, allow_pickle=True)


    dset_shape = (scores.shape[0], nf, n_prfs_total)
    
    print('final size of features for this prf:')
    print(scores.shape) 
    print('final size of whole dataset to save:')
    print(dset_shape)    
    
    print('saving to %s'%filename_save_pca)

    t = time.time()
    
    if prf_save_ind==0:
        mode='w'
    else:
        mode='r+'
        
    with h5py.File(filename_save_pca, mode) as data_set:
        
        if prf_save_ind==0:
            if compress==True:
                dset = data_set.create_dataset("features", dset_shape, dtype=save_dtype, compression='gzip')
            else:
                dset = data_set.create_dataset("features", dset_shape, dtype=save_dtype)
        else:
            assert(np.all(data_set['/features'].shape==dset_shape))
            
        data_set['/features'][:,:,prf_save_ind] = scores
        data_set.close() 
    elapsed = time.time() - t

    print('Took %.5f sec to write file'%elapsed)
    
    
def apply_pca_oneprf(features_raw, 
                     filename_save_pca,
                     load_weights_filename,
                     prf_save_ind=0,
                     n_prfs_total=1,
                     max_pc_to_retain=100, 
                     save_dtype=np.float32, compress=True, \
                     debug=False):
    
    print('loading pre-computed pca weights from %s'%(load_weights_filename))
    w = np.load(load_weights_filename, allow_pickle=True).item()
    pca_wts = w['pca_wts']
    pca_premean = w['pca_premean']
    pca_ncomp = w['pca_ncomp']
    
    n_trials, n_feat_orig = features_raw.shape
    
    print('max_pc_to_retain:')
    print(max_pc_to_retain)
    print('n_feat_orig:')
    print(n_feat_orig)
    
    # project into pca subspace using saved wts
    feat_submean = features_raw - np.tile(pca_premean[np.newaxis,:], [features_raw.shape[0],1])
    scores = feat_submean @ pca_wts.T

    n_comp_needed = int(pca_ncomp[0])

    if n_prfs_total==1:
        scores = scores[:,0:n_comp_needed]
        nf = n_comp_needed
    else:
        scores[:, n_comp_needed:] = np.nan
        if scores.shape[1]<max_pc_to_retain:
            n_pad = max_pc_to_retain - scores.shape[1]
            print('padding array with nans for %d columns'%(n_pad))
            scores = np.concatenate([scores, np.full(shape=(n_trials, n_pad), \
                                                     fill_value=np.nan, dtype=scores.dtype)], axis=1)
        nf = max_pc_to_retain
       
    dset_shape = (scores.shape[0], nf, n_prfs_total)
    
    print('final size of features for this prf:')
    print(scores.shape) 
    print('final size of whole dataset to save:')
    print(dset_shape)
    
    print('saving to %s'%filename_save_pca)
    
    t = time.time()
    
    if prf_save_ind==0:
        mode='w'
    else:
        mode='r+'
    
    with h5py.File(filename_save_pca, mode) as data_set:
        if prf_save_ind==0:
            if compress==True:
                dset = data_set.create_dataset("features", dset_shape, dtype=save_dtype, compression='gzip')
            else:
                dset = data_set.create_dataset("features", dset_shape, dtype=save_dtype)
        else:
            assert(np.all(data_set['/features'].shape==dset_shape))
            
        data_set['/features'][:,:,prf_save_ind] = scores
        data_set.close() 
    elapsed = time.time() - t

    print('Took %.5f sec to write file'%elapsed)
    
    
    
def run_pca(subject=None, \
            image_set=None, \
            feature_type=None, \
            layer_name = None, \
            which_prf_grid=5, min_pct_var=95,\
            save_weights=False, \
            use_saved_ncomp=False, \
            rm_raw = False, \
            map_res_pix=None, \
            max_pc_to_retain=None, debug=False):
       
    if subject is not None:
                      
        if image_set is None:
            # run PCA on the features for this NSD subject
            image_set = 'S%d'%subject
            use_precomputed_weights = False
            if subject==999:
                # 999 is a code for the set of images that are independent of NSD images, 
                # not shown to any participant.
                fit_inds = np.ones((10000,),dtype=bool)
            else:            
                # training / validation data always split the same way - shared 1000 inds are validation.
                subject_df = nsd_utils.get_subj_df(subject)
                fit_inds = np.array(subject_df['shared1000']==False)
        else:
            # if both subject and image set are specified, this means we should use the weights 
            # from the NSD subject and apply them to the features from image_set
            weights_image_set = 'S%d'%subject
            use_precomputed_weights = True
            assert(save_weights==False)
    else:
        # run PCA on features from a non-NSD image set
        assert(image_set is not None)
        subject=0
        use_precomputed_weights = False
        if image_set=='floc':
            labels_file = os.path.join(default_paths.floc_image_root,'floc_image_labels.csv')
            labels = pd.read_csv(labels_file)
            fit_inds = np.ones((labels.shape[0],),dtype=bool)
        else:
            raise ValueError('image_set %s not recognized'%image_set)
   
    if feature_type=='gabor':
        
        path_to_load = default_paths.gabor_texture_feat_path
        if debug:
            path_to_save = os.path.join(default_paths.gabor_texture_feat_path, 'DEBUG','PCA')
        else:
            path_to_save = os.path.join(default_paths.gabor_texture_feat_path, 'PCA')
        
        n_ori=12; n_sf = 8;
        raw_filename = os.path.join(path_to_load, \
                     '%s_features_each_prf_%dori_%dsf_gabor_solo_nonlin_grid%d.h5py'%\
                     (image_set, n_ori, n_sf, which_prf_grid))
        pca_filename = os.path.join(path_to_save, '%s_%dori_%dsf_nonlin_PCA_grid%d.h5py'%\
                     (image_set, n_ori, n_sf, which_prf_grid))
                      
        if save_weights:
            save_weights_filename = os.path.join(path_to_save, '%s_%dori_%dsf_nonlin_PCA_weights_grid%d.npy'%\
                     (image_set, n_ori, n_sf, which_prf_grid))
        else:
            save_weights_filename = None
                      
        if use_precomputed_weights:
            pca_filename = os.path.join(path_to_save, '%s_%dori_%dsf_nonlin_PCA_wtsfrom%s_grid%d.h5py'%\
                     (image_set, n_ori, n_sf, weights_image_set, which_prf_grid))
            load_weights_filename = os.path.join(path_to_save, '%s_%dori_%dsf_nonlin_PCA_weights_grid%d.npy'%\
                     (weights_image_set, n_ori, n_sf, which_prf_grid))
                      
        zscore_before_pca = False; zgroup_labels = None;
        prf_batch_size=1500
        ncomp_filename = None
        
    elif feature_type=='sketch_tokens':
        
        path_to_load = default_paths.sketch_token_feat_path
        if debug:
            path_to_save = os.path.join(default_paths.sketch_token_feat_path, 'DEBUG','PCA')
        else:
            path_to_save = os.path.join(default_paths.sketch_token_feat_path, 'PCA')
        
        raw_filename = os.path.join(path_to_load, '%s_features_each_prf_grid%d.h5py'%(image_set, which_prf_grid))
        pca_filename = os.path.join(path_to_save, '%s_PCA_grid%d.h5py'%(image_set, which_prf_grid))
        
        if save_weights:
            save_weights_filename = os.path.join(path_to_save, '%s_PCA_weights_grid%d.npy'%\
                     (image_set, which_prf_grid))
        else:
            save_weights_filename = None
                 
        if use_precomputed_weights:
            pca_filename = os.path.join(path_to_save, '%s_PCA_wtsfrom%s_grid%d.h5py'%\
                                        (image_set, weights_image_set, which_prf_grid))
            load_weights_filename = os.path.join(path_to_save, '%s_PCA_weights_grid%d.npy'%\
                     (weights_image_set, which_prf_grid))
            
        zscore_before_pca = True;
        zgroup_labels = np.concatenate([np.zeros(shape=(1,150)), np.ones(shape=(1,1))], axis=1)
        prf_batch_size=1500
        ncomp_filename = None
        
    elif 'alexnet' in feature_type:
        
        if 'blurface' in feature_type:
            path_to_load = default_paths.alexnet_blurface_feat_path
        else:
            path_to_load = default_paths.alexnet_feat_path
            
        if debug:           
            path_to_save = os.path.join(path_to_load, 'DEBUG','PCA')
        else:
            path_to_save = os.path.join(path_to_load, 'PCA')
        
                   
        raw_filename = os.path.join(path_to_load, '%s_%s_ReLU_reflect_features_each_prf_grid%d.h5py'%\
                                 (image_set, layer_name, which_prf_grid))
        pca_filename = os.path.join(path_to_save, '%s_%s_ReLU_reflect_PCA_grid%d.h5py'%\
                                    (image_set, layer_name, which_prf_grid))
        if save_weights:
            save_weights_filename = os.path.join(path_to_save, '%s_%s_ReLU_reflect_PCA_weights_grid%d.npy'%\
                                    (image_set, layer_name, which_prf_grid))
        else:
            save_weights_filename = None
            
        if use_saved_ncomp:
            ncomp_filename = os.path.join(path_to_save, '%s_%s_ReLU_reflect_PCA_grid%d_ncomp.npy'%\
                                    (image_set, layer_name, which_prf_grid))
        else:
            ncomp_filename = None
            
        if use_precomputed_weights:
            pca_filename = os.path.join(path_to_save, '%s_%s_ReLU_reflect_PCA_wtsfrom%s_grid%d.h5py'%\
                                    (image_set, layer_name, weights_image_set, which_prf_grid))
            load_weights_filename = os.path.join(path_to_save, '%s_%s_ReLU_reflect_PCA_weights_grid%d.npy'%\
                                    (weights_image_set, layer_name, which_prf_grid))
            
        zscore_before_pca = True;
        zgroup_labels = None
        prf_batch_size=100
        
    elif 'spatcolor' in feature_type:
        
        path_to_load = default_paths.spatcolor_feat_path
        
        if debug:
            path_to_save = os.path.join(path_to_load, 'DEBUG','PCA')
            path_to_load = os.path.join(path_to_load, 'DEBUG')
        else:
            path_to_save = os.path.join(path_to_load, 'PCA')
        
        prf_batch_size=15
        n_prf_batches = int(np.ceil(1456/prf_batch_size))
        
        raw_filename = [os.path.join(path_to_load, \
                               '%s_spatcolor_res%dpix_grid%d_prfbatch%d.h5py'%(image_set, \
                                                                               map_res_pix, which_prf_grid, pb)) \
                     for pb in range(n_prf_batches)]
    
        pca_filename = os.path.join(path_to_save, '%s_spatcolor_res%dpix_PCA_grid%d.h5py'%\
                                    (image_set, map_res_pix,which_prf_grid))
        if save_weights:
            save_weights_filename = os.path.join(path_to_save, '%s_spatcolor_res%dpix_PCA_weights_grid%d.npy'%\
                                    (image_set, map_res_pix,which_prf_grid))
        else:
            save_weights_filename = None
            
        if use_saved_ncomp:
            ncomp_filename = os.path.join(path_to_save,'%s_spatcolor_res%dpix_grid%d_ncomp.npy'%\
                                    (image_set, map_res_pix,which_prf_grid))
        else:
            ncomp_filename = None
            
        if use_precomputed_weights:
            pca_filename = os.path.join(path_to_save, '%s_spatcolor_res%dpix_PCA_wtsfrom%s_grid%d.h5py'%\
                                    (image_set, map_res_pix, weights_image_set, which_prf_grid))
            load_weights_filename = os.path.join(path_to_save, '%s_spatcolor_res%dpix_PCA_weights_grid%d.npy'%\
                                    (weights_image_set, map_res_pix, which_prf_grid))
            
        zscore_before_pca = False;
        zgroup_labels = None
        # prf_batch_size=100
        
        
    elif 'resnet' in feature_type:
        
        if feature_type.split('resnet_')[1]=='clip':
            path_to_load = default_paths.clip_feat_path
        elif feature_type.split('resnet_')[1]=='blurface':
            path_to_load = default_paths.resnet50_blurface_feat_path
        elif feature_type.split('resnet_')[1]=='imgnet':
            path_to_load = default_paths.resnet50_feat_path

        if debug:
            path_to_save = os.path.join(path_to_load, 'DEBUG','PCA')
            path_to_load = os.path.join(path_to_load, 'DEBUG')
        else:
            path_to_save = os.path.join(path_to_load, 'PCA')
        
        model_architecture = 'RN50'
        n_prf_batches = 15
        raw_filename = [os.path.join(path_to_load, '%s_%s_%s_features_each_prf_grid%d_prfbatch%d.h5py'%\
                                 (image_set, model_architecture,layer_name, which_prf_grid, bb)) \
                                    for bb in range(n_prf_batches)]
        pca_filename = os.path.join(path_to_save, '%s_%s_%s_PCA_grid%d.h5py'%\
                                    (image_set, model_architecture, layer_name, which_prf_grid))
        if save_weights:
            save_weights_filename = os.path.join(path_to_save, '%s_%s_%s_PCA_weights_grid%d.npy'%\
                                    (image_set, model_architecture, layer_name, which_prf_grid))
        else:
            save_weights_filename = None
            
        if use_saved_ncomp:
            ncomp_filename = os.path.join(path_to_save,'%s_%s_%s_PCA_grid%d_ncomp.npy'%\
                                    (image_set, model_architecture, layer_name, which_prf_grid))
        else:
            ncomp_filename = None
            
        if use_precomputed_weights:
            pca_filename = os.path.join(path_to_save, '%s_%s_%s_PCA_wtsfrom%s_grid%d.h5py'%\
                                    (image_set, model_architecture, layer_name, weights_image_set, which_prf_grid))
            load_weights_filename = os.path.join(path_to_save, '%s_%s_%s_PCA_weights_grid%d.npy'%\
                                    (weights_image_set, model_architecture, layer_name, which_prf_grid))
            
        zscore_before_pca = True;
        zgroup_labels = None
        prf_batch_size=100
        
    else:
        raise ValueError('feature type %s not recognized'%feature_type)
   
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
        
    if use_precomputed_weights:
        apply_pca_each_prf(raw_filename, pca_filename, \
                load_weights_filename, \
                prf_batch_size=prf_batch_size, 
                zscore_before_pca = zscore_before_pca, \
                zgroup_labels = zgroup_labels,\
                save_dtype=np.float32, compress=True, \
                debug=debug)

    else:
        run_pca_each_prf(raw_filename, pca_filename, \
                fit_inds = fit_inds, \
                prf_batch_size=prf_batch_size, 
                zscore_before_pca = zscore_before_pca, \
                zgroup_labels = zgroup_labels,\
                min_pct_var=min_pct_var,\
                max_pc_to_retain=max_pc_to_retain,\
                save_weights = save_weights, \
                save_weights_filename = save_weights_filename, \
                save_dtype=np.float32, compress=True, \
                use_saved_ncomp = use_saved_ncomp, \
                ncomp_filename = ncomp_filename, \
                debug=debug)

    if rm_raw:
        
        print('removing raw file from %s'%raw_filename)
        os.remove(raw_filename)
        print('done removing')
        
def compute_pca(values, max_pc_to_retain=None, copy_data=False):
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
    pca = decomposition.PCA(n_components = n_comp, copy=copy_data)
    scores = pca.fit_transform(values)           
    elapsed = time.time() - t
    print('Time elapsed: %.5f'%elapsed)
    values = None            
    wts = pca.components_
    ev = pca.explained_variance_
    ev = ev/np.sum(ev)*100
    pre_mean = pca.mean_
  
    return scores, wts, pre_mean, ev


def get_ncomp(pca_features_file, save_ncomp_file):
    """
    A function to extract the number of components needed for each pRF, from a saved
    PCA features file. This is for reproducibility - if we run the feature extraction procedure
    multiple times, even though the results are close, there can sometimes be a different 
    num components needed to explain 95% var. If this num changes and we 
    try to reuse the old encoding model weights, the code will break, so this is a workaround
    (enforce using the old num components when running PCA).
    
    """
    with h5py.File(pca_features_file,'r') as file:  
        dat = file['/features'][0,:,:]
        file.close()
        
    ncomp_max = dat.shape[0]
    n_prfs = dat.shape[1]
    
    wherenan = [np.where(np.isnan(dat[:,mm])) for mm in range(n_prfs)]
    ncomp = np.array([nn[0][0] if len(nn[0])>0 else ncomp_max for nn in wherenan] )

    print('saving to %s'%save_ncomp_file)
    np.save(save_ncomp_file, ncomp)
    
    return


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=0,
                    help="number of the subject, 1-8")
    parser.add_argument("--image_set", type=str,default='none',
                    help="name of the image set to use (if not an NSD subject)")
    parser.add_argument("--start_layer", type=int,default=0,
                    help="which network layer to start from?")
    parser.add_argument("--type", type=str,default='sketch_tokens',
                    help="what kind of features are we using?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--rm_raw", type=int,default=0,
                    help="want to remove raw files after PCA? 1 for yes, 0 for no")

    parser.add_argument("--save_weights", type=int,default=0,
                    help="want to save the weights to reproduce the pca later? 1 for yes, 0 for no")
    parser.add_argument("--use_saved_ncomp", type=int,default=0,
                    help="want to use a previously saved number of components? 1 for yes, 0 for no")

    parser.add_argument("--max_pc_to_retain", type=int,default=0,
                    help="max pc to retain? enter 0 for None")
    parser.add_argument("--min_pct_var", type=int,default=95,
                    help="min pct var to explain? default 95")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which pRF grid to use?")
    
    
    args = parser.parse_args()
    
    if args.subject==0:
        args.subject=None
    if args.image_set=='none':
        args.image_set=None
    
    if args.max_pc_to_retain==0:
        args.max_pc_to_retain = None
    
    if 'alexnet' in args.type:
        layers = ['Conv%d'%(ll+1) for ll in range(5)]
        layers = layers[args.start_layer:]
        for layer in layers:
            run_pca(subject=args.subject, 
                    image_set = args.image_set,\
                            feature_type=args.type, \
                            layer_name = layer,
                            which_prf_grid=args.which_prf_grid,
                            min_pct_var=args.min_pct_var, 
                            max_pc_to_retain=args.max_pc_to_retain,
                            save_weights = args.save_weights==1, 
                            use_saved_ncomp = args.use_saved_ncomp==1, 
                            rm_raw = args.rm_raw==1, 
                            debug=args.debug==1)
    else:
        run_pca(subject=args.subject, 
                image_set = args.image_set,\
                            feature_type=args.type, \
                            which_prf_grid=args.which_prf_grid,
                            min_pct_var=args.min_pct_var, 
                            max_pc_to_retain=args.max_pc_to_retain, 
                            save_weights = args.save_weights==1, 
                            use_saved_ncomp = args.use_saved_ncomp==1, 
                            debug=args.debug==1)
    