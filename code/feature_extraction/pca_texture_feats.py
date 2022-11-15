import sys, os
import numpy as np
import time, h5py

from utils import default_paths, nsd_utils
from model_fitting import initialize_fitting 
from feature_extraction import texture_feature_utils, pca_feats, fwrf_features
compute_pca = pca_feats.compute_pca

import argparse
import pandas as pd

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


"""
Code to perform PCA on features within a given feature space (texture or contour etc).
PCA is done separately within each pRF position, and the results for all pRFs are saved in a single file.
"""

def run_pca_texture_pyramid(subject=None,
                            image_set=None,
                            pca_type='pcaHL',\
                            save_weights=False, \
                            min_pct_var=95, max_pc_to_retain=150, \
                            debug=False, which_prf_grid=1, \
                            save_dtype=np.float32, compress=True):

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
            if image_set=='floc':
                labels_file = os.path.join(default_paths.floc_image_root,'floc_image_labels.csv')
                labels = pd.read_csv(labels_file)
                fit_inds = np.ones((labels.shape[0],),dtype=bool)
            else:
                raise ValueError('image_set %s not recognized'%image_set)
            
    else:
        # run PCA on features from a non-NSD image set
        assert(image_set is not None)
        use_precomputed_weights = False
        subject=0
        if image_set=='floc':
            labels_file = os.path.join(default_paths.floc_image_root,'floc_image_labels.csv')
            labels = pd.read_csv(labels_file)
            fit_inds = np.ones((labels.shape[0],),dtype=bool)
        else:
            raise ValueError('image_set %s not recognized'%image_set)
   
    n_trials = len(fit_inds)
    
    
    
    path_raw = default_paths.pyramid_texture_feat_path
    if debug:
        path_raw = os.path.join(path_raw, 'DEBUG')
    path_pca = os.path.join(path_raw, 'PCA')
    if not os.path.exists(path_pca):
        os.makedirs(path_pca)
        
    n_ori=4; n_sf=4;    
    if debug:
        prf_batch_size=2
    else:
        prf_batch_size=100;
        
    if image_set is not None:
        floader_sub=None
    else:
        floader_sub=subject
    print('floader_sub: %s'%floader_sub)
    print(image_set, subject)
    floader = fwrf_features.fwrf_feature_loader(subject=floader_sub, \
                                               image_set=image_set, \
                                               which_prf_grid=which_prf_grid, 
                                               feature_type='pyramid_texture', 
                                               n_ori=n_ori, n_sf=n_sf, 
                                               pca_type=None, 
                                               prf_batch_size=prf_batch_size)
    features_file = floader.features_file

    # batching prfs for loading, because it is a bit faster
    models = initialize_fitting.get_prf_models(which_grid = which_prf_grid)  
    n_prfs = models.shape[0]

    # can do the pca a few different ways, depending on how we want to group the features together.
    if pca_type=='pcaHL':

        feature_column_labels, feature_type_names = texture_feature_utils.get_feature_inds()
        # do higher-level feats only
        gets_pca = np.arange(5,14)

    elif pca_type=='pcaAll':

        feature_column_labels, feature_type_names = texture_feature_utils.get_feature_inds()
        # do all features
        gets_pca = np.arange(14)

    elif pca_type=='pcaHL_simple':

        feature_column_labels, feature_type_names = texture_feature_utils.get_feature_inds_simplegroups()
        # do higher-level feats only
        gets_pca = np.arange(4,10)

    elif pca_type=='pcaHL_sepscales':

        feature_column_labels, feature_type_names = texture_feature_utils.get_feature_inds_sepscales()
        # do higher-level feats only
        gets_pca = np.arange(15,41) 

    feature_type_names = np.array(feature_type_names)
    feature_type_dims = np.array([np.sum(feature_column_labels==ff) for ff in range(len(feature_type_names))])

    print('will perform PCA on these sets of features:')
    print(feature_type_names[gets_pca])
    print(feature_type_dims[gets_pca])

    feature_type_names_pca = feature_type_names[gets_pca]
    feature_inds = np.array([feature_column_labels==fi for fi in gets_pca])

    max_pc_to_retain_overall = max_pc_to_retain
    
    pca_filename_list = []
  
    # going to loop over one set of features at a time
    # (this isn't the fastest way but uses less memory at a time)
    for fi, feature_type_name in enumerate(feature_type_names_pca):
   
        n_feat_orig = np.sum(feature_inds[fi,:])
        if max_pc_to_retain_overall is None:
            max_pc_to_retain = n_feat_orig
        else:
            max_pc_to_retain = np.minimum(max_pc_to_retain_overall, n_feat_orig)

        print('\nProcessing feature subset: %s\n'%feature_type_name)
        scores_each_prf = np.zeros((n_trials, max_pc_to_retain, n_prfs), dtype=save_dtype)
        actual_max_ncomp = 0;        
        
        if use_precomputed_weights:
            wts_dir = os.path.join(path_pca, 'weights')
            load_weights_filename = os.path.join(wts_dir,'%s_%dori_%dsf_PCA_%s_only_weights_grid%d.npy'\
                               %(weights_image_set,n_ori, n_sf, feature_type_name, which_prf_grid))
            print('loading pre-computed pca weights from %s'%(load_weights_filename))
            w = np.load(load_weights_filename, allow_pickle=True).item()
            pca_wts = w['pca_wts']
            pca_premean = w['pca_premean']
            pca_ncomp = w['pca_ncomp']
            filename_save = os.path.join(path_pca, '%s_%dori_%dsf_PCA_%s_only_wtsfrom%s_grid%d.h5py'\
                               %(image_set,n_ori, n_sf, feature_type_name, weights_image_set, which_prf_grid))
        else:
            filename_save = os.path.join(path_pca, '%s_%dori_%dsf_PCA_%s_only_grid%d.h5py'\
                               %(image_set,n_ori, n_sf, feature_type_name, which_prf_grid))
        if save_weights:
            
            pca_wts = np.zeros((n_feat_orig, max_pc_to_retain, n_prfs), dtype=save_dtype)
            pca_premean = np.zeros((n_feat_orig, n_prfs), dtype=save_dtype)
            pca_ncomp = np.zeros((n_prfs,),dtype=save_dtype)
            
        for prf_model_index in range(n_prfs):

            if debug and prf_model_index>1:
                continue
                
            print('Processing pRF %d of %d'%(prf_model_index, n_prfs))
            sys.stdout.flush()
            
            # load raw features
            feat, _ = floader.load(np.arange(n_trials), prf_model_index)

            # pull out the ones for this "type" only
            features_in_prf = feat[:,feature_inds[fi,:]].astype(np.float32)
            
            print('Processing %s, size of array before PCA:'%feature_type_name)
            print(features_in_prf.shape)

            if use_precomputed_weights:
                
                # project into pca subspace using saved wts
                wts = pca_wts[:,:,prf_model_index]
                pre_mean = pca_premean[:,prf_model_index]
                n_comp_needed = int(pca_ncomp[prf_model_index])
                
                feat_submean = features_in_prf - np.tile(pre_mean[np.newaxis,:], [features_in_prf.shape[0],1])
                scores = feat_submean @ wts
                
            else:
                
                # compute pca weights here
                _, wts, pre_mean, ev = compute_pca(features_in_prf[fit_inds,:], max_pc_to_retain=max_pc_to_retain)
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
            
            if save_weights:
                pca_wts[:,:,prf_model_index] = wts.T
                pca_premean[:,prf_model_index] = pre_mean
                pca_ncomp[prf_model_index] = n_comp_needed

        # To save space, get rid of portion of array that ended up all nans
        if debug:
            actual_max_ncomp=np.max([2,actual_max_ncomp])
            assert(np.all((scores_each_prf[:,actual_max_ncomp:,:]==0) | \
                          np.isnan(scores_each_prf[:,actual_max_ncomp:,:])))
        else:
            assert(np.all(np.isnan(scores_each_prf[:,actual_max_ncomp:,:])))
        scores_each_prf = scores_each_prf[:,0:actual_max_ncomp,:]
        print('final size of array to save (for %s):'%feature_type_name)
        print(scores_each_prf.shape)
        
        
        print('saving to %s'%filename_save)
        pca_filename_list.append(filename_save)
        sys.stdout.flush()
        t = time.time()
        with h5py.File(filename_save, 'w') as data_set:
            if compress==True:
                dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype, compression='gzip')
            else:
                dset = data_set.create_dataset("features", np.shape(scores_each_prf), dtype=save_dtype)
            data_set['/features'][:,:,:] = scores_each_prf
            data_set.close() 
        elapsed = time.time() - t
        print('Took %.5f sec to write file'%elapsed)
        sys.stdout.flush()
        
        if save_weights:
            wts_dir = os.path.join(path_pca, 'weights')
            if not os.path.exists(wts_dir):
                os.makedirs(wts_dir)
            save_weights_filename = os.path.join(wts_dir,'%s_%dori_%dsf_PCA_%s_only_weights_grid%d.npy'\
                               %(image_set,n_ori, n_sf, feature_type_name, which_prf_grid))
            wts = {'pca_wts': pca_wts, 'pca_premean': pca_premean, 'pca_ncomp': pca_ncomp}
            print('saving the weights for this pca to %s'%save_weights_filename)
            np.save(save_weights_filename, wts, allow_pickle=True)
        
        
    # Now going to concatenate all files together, for easier loading later
    if use_precomputed_weights:
        concat_filename = os.path.join(path_pca, '%s_%dori_%dsf_%s_concat_wtsfrom%s_grid%d.h5py'\
                                   %(image_set, n_ori, n_sf, pca_type, weights_image_set, which_prf_grid))
        concat_labels_filename = os.path.join(path_pca, '%s_%dori_%dsf_featurelabels_%s_wtsfrom%s_grid%d.npy'\
                                   %(image_set, n_ori, n_sf, pca_type, weights_image_set, which_prf_grid))
    else:
        concat_filename = os.path.join(path_pca, '%s_%dori_%dsf_%s_concat_grid%d.h5py'\
                                       %(image_set, n_ori, n_sf, pca_type, which_prf_grid))
        concat_labels_filename = os.path.join(path_pca, '%s_%dori_%dsf_featurelabels_%s_grid%d.npy'\
                                       %(image_set, n_ori, n_sf, pca_type, which_prf_grid))

    
    # first grab the raw low-level features (if using)
    if pca_type is not 'pcaAll':
        is_ll = texture_feature_utils.is_low_level()
        print('loading from %s'%features_file)
        st = time.time()
        sys.stdout.flush()
        with h5py.File(features_file, 'r') as file:
            feat_raw = np.array(file['/features'])[:,is_ll,:]
            feat_shape = np.shape(file['/features'])
            file.close()
        elapsed = time.time() - st
        print('loading took %.5f sec'%elapsed)
        print(feat_raw.shape)
        feat_all = feat_raw;
        sys.stdout.flush()
        
        feature_column_labels_pca = np.array(feature_column_labels[is_ll])
    else:
        feat_raw = None       
        feature_column_labels_pca = np.array([])
        

    n_raw_types = len(np.unique(feature_column_labels_pca))

    # loop over the higher-level feature subsets
    for fi, pca_filename in enumerate(pca_filename_list):

        print('loading from %s'%pca_filename)
        st = time.time()
        sys.stdout.flush()
        with h5py.File(pca_filename, 'r') as file:
            feat_pca = np.array(file['/features'])
            file.close()
        elapsed = time.time() - st
        print('loading took %.5f sec'%elapsed)
        print(feat_pca.shape)
        sys.stdout.flush()
        n_pc = feat_pca.shape[1]

        if feat_raw is None and fi==0:
            feat_all = feat_pca
        else:
            feat_all = np.concatenate([feat_all, feat_pca], axis=1)

        feature_column_labels_pca = np.concatenate([feature_column_labels_pca, \
                                                    (fi+n_raw_types)*np.ones((n_pc,))], axis=0)

    print('final shape is:')
    print(feat_all.shape)
    
    print('saving to %s'%concat_filename)
    sys.stdout.flush()
    t = time.time()
    with h5py.File(concat_filename, 'w') as data_set:
        if compress==True:
            dset = data_set.create_dataset("features", np.shape(feat_all), dtype=save_dtype, compression='gzip')
        else:
            dset = data_set.create_dataset("features", np.shape(feat_all), dtype=save_dtype)
        data_set['/features'][:,:,:] = feat_all
        data_set.close() 
    elapsed = time.time() - t

    print('Took %.5f sec to write file'%elapsed)

    # save the labels for which columns of the concatenated array correspond to which feature types
    print('saving to %s'%concat_labels_filename)
    print(feature_column_labels_pca, feature_type_names)
    np.save(concat_labels_filename, {'feature_column_labels': feature_column_labels_pca, \
                             'feature_type_names': feature_type_names})

    
    # remove the smaller intermediate files, to save disk space  
    for pca_filename in pca_filename_list:
        
        print('deleting %s'%pca_filename)
        if not debug:
            os.remove(pca_filename)
            
            
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=0,
                    help="number of the subject, 1-8")
    parser.add_argument("--image_set", type=str,default='none',
                    help="name of the image set to use (if not an NSD subject)")
    
    parser.add_argument("--pca_type", type=str,default='pcaHL',
                    help="what kind of features are we using?")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")

    parser.add_argument("--max_pc_to_retain", type=int,default=0,
                    help="max pc to retain? enter 0 for None")
    parser.add_argument("--min_pct_var", type=int,default=95,
                    help="min pct var to explain? default 95")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which pRF grid to use?")
    
    parser.add_argument("--save_weights", type=int,default=0,
                    help="want to save the weights to reproduce the pca later? 1 for yes, 0 for no")

    
    args = parser.parse_args()
    
    if args.subject==0:
        args.subject=None
    if args.image_set=='none':
        args.image_set=None
        
    if args.max_pc_to_retain==0:
        args.max_pc_to_retain = None
     
    run_pca_texture_pyramid(subject=args.subject, 
                            image_set = args.image_set,
                            pca_type=args.pca_type, \
                            min_pct_var=args.min_pct_var, \
                            max_pc_to_retain=args.max_pc_to_retain,\
                            debug=args.debug==1,  \
                            which_prf_grid=args.which_prf_grid, \
                            save_weights = args.save_weights)
    