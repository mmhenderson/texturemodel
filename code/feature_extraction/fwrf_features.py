import numpy as np
import os
import time
import h5py
import copy

from utils import default_paths
from model_fitting import initialize_fitting

"""
Code to load pre-computed features from various models (gabor, alexnet, semantic, etc.)
Features have been computed at various spatial locations in a grid (pRF positions/sizes)
"""
          
class fwrf_feature_loader:
    
    def __init__(self, subject, which_prf_grid, feature_type, **kwargs):
        
        self.subject = subject            
        self.which_prf_grid = which_prf_grid
        self.__init_prf_batches__(kwargs)        
        self.feature_type = feature_type
        self.include_solo_models = kwargs['include_solo_models'] \
            if 'include_solo_models' in kwargs.keys() else False   
        
        if self.feature_type=='gabor_solo':
            self.__init_gabor_solo__(kwargs)
        elif self.feature_type=='sketch_tokens':
            self.__init_sketch_tokens__(kwargs)
        elif self.feature_type=='pyramid_texture':
            self.__init_pyramid_texture__(kwargs)
        elif self.feature_type=='alexnet':
            self.__init_alexnet__(kwargs)
        elif self.feature_type=='clip':
            self.__init_clip__(kwargs)
        else:
            raise ValueError('feature type %s not recognized')

        if not os.path.exists(self.features_file):
            raise RuntimeError('Looking at %s for precomputed features, not found.'%self.features_file)

    def __init_gabor_solo__(self, kwargs):
        
        self.use_pca_feats = kwargs['use_pca_feats'] if 'use_pca_feats' in kwargs.keys() else False        
        self.n_ori = kwargs['n_ori'] if 'n_ori' in kwargs.keys() else 12
        self.n_sf = kwargs['n_sf'] if 'n_sf' in kwargs.keys() else 8
        self.nonlin_fn = kwargs['nonlin_fn'] if 'nonlin_fn' in kwargs.keys() else True
        
        gabor_texture_feat_path = default_paths.gabor_texture_feat_path
        if self.use_pca_feats:
            assert(self.nonlin_fn==True)
            self.features_file = os.path.join(gabor_texture_feat_path, 'PCA', \
                                          'S%d_%dori_%dsf_nonlin_PCA_grid%d.h5py'\
                                           %(self.subject, self.n_ori, self.n_sf, self.which_prf_grid))  
            with h5py.File(self.features_file, 'r') as file:
                feat_shape = np.shape(file['/features'])
                file.close()
            self.max_features = feat_shape[1]
        else:
            self.features_file = os.path.join(gabor_texture_feat_path, \
                      'S%d_features_each_prf_%dori_%dsf_gabor_solo'%(self.subject, self.n_ori, self.n_sf))
            self.max_features = self.n_ori*self.n_sf             
            if self.nonlin_fn:
                self.features_file += '_nonlin'
            self.features_file += '_grid%d.h5py'%self.which_prf_grid                                             

        self.do_varpart = False
        self.n_feature_types=1
        
    def __init_pyramid_texture__(self, kwargs):
                
        from feature_extraction import texture_feature_utils
        pyramid_texture_feat_path = default_paths.pyramid_texture_feat_path
        self.do_varpart=kwargs['do_varpart'] if 'do_varpart' in kwargs.keys() else True
        self.n_ori = kwargs['n_ori'] if 'n_ori' in kwargs.keys() else 4
        self.n_sf = kwargs['n_sf'] if 'n_sf' in kwargs.keys() else 4
        self.pca_type=kwargs['pca_type'] if 'pca_type' in kwargs.keys() else None
        
        self.group_all_hl_feats=kwargs['group_all_hl_feats'] if 'group_all_hl_feats' in kwargs.keys() else True
        
        if self.pca_type is not None:
            self.use_pca_feats=True
            # will use features where higher-level sub-sets have been reduced in dim with PCA.
            self.features_file = os.path.join(pyramid_texture_feat_path, 'PCA', \
                                              'S%d_%dori_%dsf_%s_concat_grid%d.h5py'%\
                                              (self.subject, self.n_ori, self.n_sf, \
                                               self.pca_type, self.which_prf_grid))
            self.feature_column_labels, self.feature_type_names = \
                texture_feature_utils.get_feature_inds_pca(self.subject, self.pca_type, self.which_prf_grid)
        else:
            self.use_pca_feats=False
            # will load from raw array (641 features).
            self.features_file = os.path.join(pyramid_texture_feat_path, \
                                              'S%d_features_each_prf_%dori_%dsf_grid%d.h5py'%\
                                              (self.subject, self.n_ori, self.n_sf, self.which_prf_grid))
            self.feature_column_labels, self.feature_type_names = \
                texture_feature_utils.get_feature_inds_simplegroups();
        
        self.max_features = len(self.feature_column_labels)
        
        # when fitting ridge regression model, which features have same corresponding lambda?
        n_ll_feats = 4;
        self.feature_groups_ridge = (self.feature_column_labels>=n_ll_feats).astype('int')
        
        # when doing variance partition, which features go together?
        if self.group_all_hl_feats:
            # group the first 4 and last 6 sets of features.   
            self.feature_column_labels = (self.feature_column_labels>=n_ll_feats).astype('int')
            self.feature_group_names = ['lower-level', 'higher-level']
        else:
            # treat all 10 sets as separate groups
            self.feature_column_labels = self.feature_column_labels
            self.feature_group_names = self.feature_type_names
        self.n_feature_types = len(self.feature_group_names)
       
        if not os.path.exists(self.features_file):
            raise RuntimeError('Looking at %s for precomputed features, not found.'%self.features_file)  

        
      
    def __init_sketch_tokens__(self, kwargs):

        sketch_token_feat_path = default_paths.sketch_token_feat_path

        self.use_pca_feats = kwargs['use_pca_feats'] if 'use_pca_feats' in kwargs.keys() else False
        self.use_residual_st_feats = kwargs['use_residual_st_feats'] \
                                        if 'use_residual_st_feats' in kwargs.keys() else False
           
        if self.use_pca_feats:
            self.features_file = os.path.join(sketch_token_feat_path, 'PCA', \
                                          'S%d_PCA_grid%d.h5py'%(self.subject, self.which_prf_grid))  
            with h5py.File(self.features_file, 'r') as file:
                feat_shape = np.shape(file['/features'])
                file.close()
            self.max_features = feat_shape[1]
        elif self.use_residual_st_feats:
            self.max_features = 150;
            self.features_file = os.path.join(sketch_token_feat_path, \
                          'S%d_gabor_residuals_grid%d.h5py'%(self.subject, self.which_prf_grid))
             
        else:
            self.max_features = 150;
            self.features_file = os.path.join(sketch_token_feat_path, \
                          'S%d_features_each_prf_grid%d.h5py'%(self.subject, self.which_prf_grid))
            
        self.do_varpart=False
        self.n_feature_types=1
    
    def __init_alexnet__(self,kwargs):
    
        from feature_extraction import extract_alexnet_features
        self.padding_mode = kwargs['padding_mode'] if 'padding_mode' in kwargs.keys() else 'reflect'
        if 'layer_name' not in kwargs.keys():
            raise ValueError('for alexnet, need to specify a layer name')
        self.layer_name = kwargs['layer_name']
        self.use_pca_feats = kwargs['use_pca_feats'] if 'use_pca_feats' in kwargs.keys() else True 
        
        alexnet_feat_path = default_paths.alexnet_feat_path
        alexnet_layer_names  = extract_alexnet_features.alexnet_layer_names
        n_features_each_layer = extract_alexnet_features.n_features_each_layer

        if self.use_pca_feats:        
            self.features_file = os.path.join(alexnet_feat_path, 'PCA', \
              'S%d_%s_reflect_PCA_grid%d.h5py'%(self.subject, self.layer_name, self.which_prf_grid))
        else:
            raise RuntimeError('have to use pca feats for alexnet') 

        layer_ind = [ll for ll in range(len(alexnet_layer_names)) \
                         if alexnet_layer_names[ll]==self.layer_name]
        assert(len(layer_ind)==1)
        layer_ind = layer_ind[0]        
        n_feat_expected = n_features_each_layer[layer_ind]

        with h5py.File(self.features_file, 'r') as file:
            feat_shape = np.shape(file['/features'])
            file.close()
        self.max_features = feat_shape[1]
        
        self.do_varpart=False
        self.n_feature_types=1
        
    def __init_clip__(self, kwargs):
        
        from feature_extraction import extract_clip_features
        self.model_architecture = kwargs['model_architecture'] if 'model_architecture' in kwargs.keys() else 'RN50'
        if 'layer_name' not in kwargs.keys():
            raise ValueError('for clip, need to specify a layer name')
        self.layer_name = kwargs['layer_name']
        self.use_pca_feats = kwargs['use_pca_feats'] if 'use_pca_feats' in kwargs.keys() else True 
        
        clip_feat_path = default_paths.clip_feat_path
        clip_layer_names  = extract_clip_features.resnet_block_names
        n_features_each_layer = extract_clip_features.n_features_each_resnet_block

        if self.use_pca_feats:
            self.features_file = os.path.join(clip_feat_path, 'PCA', \
              'S%d_%s_%s_PCA_grid%d.h5py'%(self.subject, self.model_architecture, \
                                         self.layer_name, self.which_prf_grid))  
        else:
            raise RuntimeError('have to use pca feats for clip')
            
        layer_ind = [ll for ll in range(len(clip_layer_names)) \
                         if clip_layer_names[ll]==self.layer_name]
        assert(len(layer_ind)==1)
        layer_ind = layer_ind[0]        
        n_feat_expected = n_features_each_layer[layer_ind]
           
        with h5py.File(self.features_file, 'r') as file:
            feat_shape = np.shape(file['/features'])
            file.close()
        self.max_features = feat_shape[1]
        
        self.do_varpart=False
        self.n_feature_types=1
    
    def __init_prf_batches__(self, kwargs):
        
        self.prf_batch_size = kwargs['prf_batch_size'] if 'prf_batch_size' in kwargs.keys() else 100
        self.n_prfs = initialize_fitting.get_prf_models(which_grid=self.which_prf_grid).shape[0]
        n_prf_batches = int(np.ceil(self.n_prfs/self.prf_batch_size))          
        self.prf_batch_inds = [np.arange(self.prf_batch_size*bb, np.min([self.prf_batch_size*(bb+1), self.n_prfs])) \
                               for bb in range(n_prf_batches)]
       
        self.features_each_prf_batch = None
        self.prf_inds_loaded = []
        
    def clear_big_features(self):
        
        print('Clearing features from memory')
        self.features_each_prf_batch = None 
        self.prf_inds_loaded = []
        if hasattr(self,'is_defined_each_prf_batch'):
            self.is_defined_each_prf_batch = None
        
    def get_partial_versions(self):

        partial_version_names = ['full_model']
        masks = np.ones([1,self.max_features])
        
        if self.do_varpart and self.n_feature_types>1:

            if (self.n_feature_types<=2) or self.include_solo_models:
                # "Partial versions" will be listed as: [full model, model w only first set of features,
                # model w only second set, ...             
                partial_version_names += ['just_%s'%ff for ff in self.feature_group_names]
                masks2 = np.concatenate([np.expand_dims(np.array(self.feature_column_labels==ff).astype('int'), axis=0) \
                                         for ff in np.arange(0,self.n_feature_types)], axis=0)
                masks = np.concatenate((masks, masks2), axis=0)

            if self.n_feature_types > 2:
                # if more than two types, also include models where we leave out first set of features, 
                # leave out second set of features...]
                partial_version_names += ['leave_out_%s'%ff for ff in self.feature_group_names]           
                masks3 = np.concatenate([np.expand_dims(np.array(self.feature_column_labels!=ff).astype('int'), axis=0) \
                                         for ff in np.arange(0,self.n_feature_types)], axis=0)
                masks = np.concatenate((masks, masks3), axis=0)           

        return masks, partial_version_names

    def get_feature_group_inds(self):
        
        if hasattr(self, 'feature_groups_ridge'):
            group_inds = self.feature_groups_ridge
        else:
            group_inds = np.zeros((self.max_features,),dtype=int)
            
        return group_inds
        
    def __load_features_prf_batch__(self, image_inds, prf_model_index):
        
        # loading features for pRFs in batches to speed up a little
        t = time.time()
        batch_to_use = np.where([prf_model_index in self.prf_batch_inds[bb] for \
                                     bb in range(len(self.prf_batch_inds))])[0][0]
        assert(prf_model_index in self.prf_batch_inds[batch_to_use])
        self.prf_inds_loaded = self.prf_batch_inds[batch_to_use]
        print('Loading pre-computed features for models [%d - %d] from %s'%\
               (self.prf_batch_inds[batch_to_use][0], self.prf_batch_inds[batch_to_use][-1], \
                self.features_file))
        self.features_each_prf_batch = None

        t = time.time()
        with h5py.File(self.features_file, 'r') as data_set:
            values = np.copy(data_set['/features'][:,:,self.prf_batch_inds[batch_to_use]])
            data_set.close() 
        elapsed = time.time() - t
        print('Took %.5f seconds to load file'%elapsed)

        self.features_each_prf_batch = values[image_inds,:,:]
        self.features_each_prf_batch = self.features_each_prf_batch[:,0:self.max_features,:]
        values = None
        print('Size of features array for this image set and batch is:')
        print(self.features_each_prf_batch.shape)

        if self.use_pca_feats:

            # if the features have been reduced with PCA, then they will have different dimension
            # for different pRFs. The remaining values are filled in with nans, so we need to find 
            # the non-nan values here.
            nan_inds = [np.isnan(self.features_each_prf_batch[0,:,mm]) \
                        for mm in range(len(self.prf_batch_inds[batch_to_use]))]
            self.is_defined_each_prf_batch = [~n for n in nan_inds]
                
            print('Number of nans in each prf this batch:')
            print([np.sum(n) for n in nan_inds])
            
    
    def load(self, image_inds, prf_model_index):
         
        if image_inds.dtype=='bool' or np.all(np.isin(np.unique(image_inds),[0,1])):
            print('\nWARNING: image_inds (len %d) looks like a boolean array'%len(image_inds))
            print('you might need to do np.where(image_inds) first\n')
            
        if prf_model_index not in self.prf_inds_loaded:            
            self.__load_features_prf_batch__(image_inds, prf_model_index)
        
        # get features for the current pRF from the loaded batch 
        index_into_batch = np.where(prf_model_index==self.prf_inds_loaded)[0][0]
        print('Index into batch for prf %d: %d'%(prf_model_index, index_into_batch))
        
        features_in_prf = self.features_each_prf_batch[:,:,index_into_batch]            
       
        assert(features_in_prf.shape[0]==len(image_inds))
        assert(features_in_prf.shape[1]==self.max_features)
        
        if hasattr(self,'is_defined_each_prf_batch'):
            feature_inds_defined = self.is_defined_each_prf_batch[index_into_batch]
            features_in_prf = features_in_prf[:,feature_inds_defined]
            assert(not np.any(np.isnan(features_in_prf)))
        else:
            feature_inds_defined = np.ones((self.max_features,), dtype=bool)
                    
        if np.any(np.sum(features_in_prf, axis=0)==0):
            print('Warning: there are columns of all zeros in features matrix, columns:')
            print(np.where(np.sum(features_in_prf, axis=0)==0))
        
        print('Final size of feature matrix for this image set and pRF is:')
        print(features_in_prf.shape)
        
        return features_in_prf, feature_inds_defined
    

def get_features_each_prf(image_inds, feature_loader, \
                          zscore=False, debug=False, \
                          dtype=np.float32):
    
    """ 
    Just loads the features in each pRF on each trial, and optionally z-score.
    Returns [trials x features x pRFs]
    """
    
    n_trials = len(image_inds)
    
    n_features_max = feature_loader.max_features
   
    n_prfs = feature_loader.n_prfs
    
    features_each_prf = np.full(fill_value=0, shape=(n_trials, n_features_max, n_prfs), dtype=dtype)
     
    feature_loader.clear_big_features()

    for mm in range(n_prfs):
        if mm>1 and debug:
            break

        # all_feat_concat is size [ntrials x nfeatures] (where nfeatures can be <max_features)
        # feature_inds_defined is [max_features]
        all_feat_concat, feature_inds_defined = feature_loader.load(image_inds, mm)

        if zscore:
            m = np.mean(all_feat_concat, axis=0)
            s = np.std(all_feat_concat, axis=0)
            all_feat_concat = (all_feat_concat - m)/s
            assert(not np.any(np.isnan(all_feat_concat)) and not np.any(np.isinf(all_feat_concat)))

        # saving all these features for use later on
        features_each_prf[:,feature_inds_defined,mm] = all_feat_concat
         
    feature_loader.clear_big_features()
                    
    return features_each_prf