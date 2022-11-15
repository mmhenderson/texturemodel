import numpy as np
import os, h5py
import torch
import time
import pandas as pd

from utils import default_paths


# Utility functions for figuring out which columns in the big texture statistics matrix include which types of features

# these are the names of all the feature types in the model (14)
feature_type_names_raw = ['pixel_stats', 'mean_magnitudes', 'mean_realparts', \
                     'marginal_stats_lowpass_recons', 'variance_highpass_resid', \
                     'magnitude_feature_autocorrs', 'lowpass_recon_autocorrs', \
                     'highpass_resid_autocorrs', \
                     'magnitude_within_scale_crosscorrs', 'real_within_scale_crosscorrs', \
                     'magnitude_across_scale_crosscorrs', 'real_imag_across_scale_crosscorrs', \
                     'real_spatshift_within_scale_crosscorrs', 'real_spatshift_across_scale_crosscorrs']
feature_type_dims_raw = [6,16,16,10,1,272,73,25,24,24,48,96,10,20]

# These are the final 10 sets of features that we used.
# They combine a few of the subsets that have similar kinds of features.
features_to_groups = np.array([0,1,2,3,3,4,5,5,6,7,8,9,7,9])
feature_type_names_simple = ['pixel','energy-mean','linear-mean','marginal',\
               'energy-auto','linear-auto',\
               'energy-cross-orient','linear-cross-orient',\
               'energy-cross-scale','linear-cross-scale']

def get_feature_inds():    

    feature_column_labels = np.concatenate([np.ones((nf,))*fi \
                            for fi, nf in enumerate(feature_type_dims_raw)], axis=0)
       
    return feature_column_labels, feature_type_names_raw

def get_feature_inds_simplegroups():

    columns_raw = get_feature_inds()[0]
    columns_new = np.zeros(np.shape(columns_raw))
    for gg in range(len(np.unique(features_to_groups))):
        feature_inds = np.where(features_to_groups==gg)[0]
        columns_new[np.isin(columns_raw,feature_inds)] = gg;
   
    return columns_new, feature_type_names_simple

def get_feature_inds_sepscales():
    
    filename = os.path.join(default_paths.pyramid_texture_feat_path, \
                           'feature_column_labels_4ori_4sf.csv')
    df = pd.read_csv(filename)
    f  = np.array(df['feature_type_raw']).astype(int)
    s = np.array(df['scale']).astype(int)

    unique_combs, feature_column_labels = np.unique(np.array([f,s]).T, axis=0, return_inverse=True)
    
    feature_type_names = ['%s_scale%d'%(feature_type_names_raw[unique_combs[ii,0]], unique_combs[ii,1]) \
         for ii in range(len(unique_combs))]

    return feature_column_labels, feature_type_names

def get_feature_inds_lowhigh():    
    
    # Numbers that define which feature types are in which columns of final output matrix
    n_total_features = np.sum(feature_type_dims_raw)
    n_ll_features = np.sum(feature_type_dims_raw[0:5])
    feature_column_labels = (np.arange(n_total_features)>=n_ll_features).astype(int)
    
    return feature_column_labels, ['lower-level', 'higher-level']

def is_low_level():
    
    is_ll = get_feature_inds_lowhigh()[0]==0
    
    return is_ll

def get_feature_inds_pca(image_set, pca_type='pcaHL', \
                         which_prf_grid=5):

    filename = os.path.join(default_paths.pyramid_texture_feat_path,'PCA', \
                            '%s_4ori_4sf_featurelabels_%s_grid%d.npy'\
                               %(image_set, pca_type, which_prf_grid))
    lab = np.load(filename, allow_pickle=True).item()
    cols = lab['feature_column_labels'].astype(int)
    names = lab['feature_type_names']
   
    if pca_type=='pcaHL_simple':
        feature_column_labels = cols;
        feature_type_names = names;
    elif pca_type=='pcaHL' or pca_type=='pcaAll':
        # need to convert indices that go 0-14 to 0-10
        columns_new = np.zeros(np.shape(cols))
        for gg in range(len(np.unique(features_to_groups))):
            feature_inds = np.where(features_to_groups==gg)[0]
            columns_new[np.isin(cols,feature_inds)] = gg;
        feature_column_labels = columns_new;
        feature_type_names  = feature_type_names_simple
    elif pca_type=='pcaHL_sepscales':
        # need to convert indices that go 0-41 to 0-10
        columns_new = np.zeros(np.shape(cols))
        for gg in range(len(np.unique(features_to_groups))):
            feature_inds = np.where(features_to_groups==gg)[0]
            feature_names = np.array(feature_type_names_raw)[feature_inds]
            inds = np.array([[fname in name for name in names] for fname in feature_names])
            inds = np.where(np.any(inds, axis=0))[0]
            columns_new[np.isin(cols, inds)] = gg;
        feature_column_labels = columns_new;
        feature_type_names  = feature_type_names_simple

    return feature_column_labels, feature_type_names


def make_feature_column_labels():

    """
    Creating a CSV file that indicates which feature types (and which scales)
    are in each column of the big features matrix
    """
    
    from feature_extraction import texture_statistics_pyramid

    # pass in some fake data, just need shapes
    images = np.random.normal(0,1,[10,1,240,240])

    device = 'cpu:0'
    
    #### Create pyramid feature extractor 
    n_ori=4;
    n_sf=4;

    _fmaps_fn = texture_statistics_pyramid.steerable_pyramid_extractor(pyr_height=n_sf, n_ori = n_ori)

    print('Running steerable pyramid feature extraction...')
    print('Images array shape is:')
    print(images.shape)
    t = time.time()
    fmaps = _fmaps_fn(images, to_torch=False, device=device)        
    elapsed =  time.time() - t
    print('time elapsed = %.5f'%elapsed)

    #### Extract all the texture features (for one example pRF)
    prf_params = [0,0,0.5]

    f = texture_statistics_pyramid.get_all_features(fmaps, images, prf_params, \
                                                    sample_batch_size=20, n_prf_sd_out=2, \
                                                    aperture=1.0, \
                                                    device=device, \
                                                    keep_orig_shape=True)
    feature_column_labels_all = []
    scale_labels_all = []

    for fi, feat in enumerate(f):

        inds_keep = torch.sum(torch.reshape(feat,[feat.shape[0],-1]), axis=0)!=0
        inds_keep = inds_keep.detach().cpu().numpy()

        if len(feat.shape)>2 and feat.shape[1]>1:
            scale_inds = np.zeros(feat.shape[1:])

            for sc in range(feat.shape[1]):
                scale_inds[sc,:] = sc

            scale_inds = torch.Tensor(scale_inds[None,:])
            scale_inds = torch.reshape(scale_inds, [1,-1])
            scale_inds = scale_inds.detach().cpu().numpy()

            scale_inds = np.squeeze(scale_inds)

        else:

            scale_inds = np.zeros((np.prod(feat.shape[1:]),))

        scale_inds = scale_inds[inds_keep]
        scale_labels_all += list(scale_inds)

        nf = len(scale_inds)
        feature_column_labels_all += list(fi*np.ones((nf,)))

    feature_column_labels_all = np.array(feature_column_labels_all).astype(int)
    scale_labels_all = np.array(scale_labels_all).astype(int)

    assert(np.all(feature_column_labels_all==get_feature_inds()[0]))
           
    # some other ways of grouping the features, which will be useful for our 
    # variance partition analyses
    feature_columns_simple = get_feature_inds_simplegroups()[0]
    is_ll = get_feature_inds_lowhigh()[0]
    
    arr = np.array([feature_column_labels_all, scale_labels_all, \
                    feature_columns_simple, is_ll]).T

    df = pd.DataFrame(arr, \
                columns=['feature_type_raw', 'scale', 'feature_type_simple', 'low_high'],)
    
    fn2save = os.path.join(default_paths.pyramid_texture_feat_path, \
                           'feature_column_labels_%dori_%dsf.csv'%(n_ori, n_sf))
    print('saving to %s'%fn2save)
    df.to_csv(fn2save, index=False)