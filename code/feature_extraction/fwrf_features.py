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
    
    def __init__(self, subject=None, image_set=None, \
                 which_prf_grid=5, feature_type='gabor_solo', \
                 **kwargs):
        
        if subject is not None: 
            self.image_set = 'S%d'%subject
        else:
            self.image_set = image_set
          
        assert(self.image_set is not None)
        print('making feature loader with image set: %s'%self.image_set)
        
        self.which_prf_grid = which_prf_grid
        self.__init_prf_batches__(kwargs)        
        self.feature_type = feature_type
        self.include_solo_models = kwargs['include_solo_models'] \
            if 'include_solo_models' in kwargs.keys() else False   
        self.pca_subject = kwargs['pca_subject'] \
            if 'pca_subject' in kwargs.keys() else None
        
        if self.feature_type=='gabor_solo':
            self.__init_gabor__(kwargs)
        elif self.feature_type=='gist':
            self.__init_gist__(kwargs)
        elif self.feature_type=='sketch_tokens':
            self.__init_sketch_tokens__(kwargs)
        elif self.feature_type=='pyramid_texture':
            self.__init_pyramid_texture__(kwargs)
        elif self.feature_type=='color':
            self.__init_color__(kwargs)
        elif self.feature_type=='alexnet':
            self.__init_alexnet__(kwargs)
        elif self.feature_type=='resnet':
            self.__init_resnet__(kwargs)
        else:
            raise ValueError('feature type %s not recognized')

        if not os.path.exists(self.features_file):
            raise RuntimeError('Looking at %s for precomputed features, not found.'%self.features_file)

    def __init_gabor__(self, kwargs):
        
        self.use_pca_feats = kwargs['use_pca_feats'] if 'use_pca_feats' in kwargs.keys() else False        
        self.n_ori = kwargs['n_ori'] if 'n_ori' in kwargs.keys() else 12
        self.n_sf = kwargs['n_sf'] if 'n_sf' in kwargs.keys() else 8
        self.nonlin_fn = kwargs['nonlin_fn'] if 'nonlin_fn' in kwargs.keys() else True
        self.use_noavg = kwargs['use_noavg'] if 'use_noavg' in kwargs.keys() else False
        if self.use_noavg:
            self.use_pca_feats=True
            
        feat_path = default_paths.gabor_texture_feat_path
        if self.use_pca_feats:
            assert(self.nonlin_fn==True)
            if self.use_noavg:
                if self.pca_subject is not None:
                    self.features_file = os.path.join(feat_path, 'PCA', \
                                            '%s_gabor_noavg_%dori_%dsf_PCA_wtsfromS%d_grid%d.h5py'%\
                                            (self.image_set, self.n_ori, self.n_sf, self.pca_subject, self.which_prf_grid))
                else:
                    self.features_file = os.path.join(feat_path, 'PCA', \
                                            '%s_gabor_noavg_%dori_%dsf_PCA_grid%d.h5py'%\
                                            (self.image_set, self.n_ori, self.n_sf, self.which_prf_grid))
            else:
                if self.pca_subject is not None:
                    self.features_file = os.path.join(feat_path, 'PCA', \
                                              '%s_%dori_%dsf_nonlin_PCA_wtsfromS%d_grid%d.h5py'\
                                               %(self.image_set, self.n_ori, self.n_sf, self.pca_subject, self.which_prf_grid))  
                else:
                    self.features_file = os.path.join(feat_path, 'PCA', \
                                              '%s_%dori_%dsf_nonlin_PCA_grid%d.h5py'\
                                               %(self.image_set, self.n_ori, self.n_sf, self.which_prf_grid))  
            with h5py.File(self.features_file, 'r') as file:
                feat_shape = np.shape(file['/features'])
                file.close()
            self.max_features = feat_shape[1]
        else:
            self.features_file = os.path.join(feat_path, \
                      '%s_features_each_prf_%dori_%dsf_gabor_solo'%(self.image_set, self.n_ori, self.n_sf))
            self.max_features = self.n_ori*self.n_sf             
            if self.nonlin_fn:
                self.features_file += '_nonlin'
            self.features_file += '_grid%d.h5py'%self.which_prf_grid                                             

        self.do_varpart = False
        self.n_feature_types=1
        
    def __init_pyramid_texture__(self, kwargs):
                
        from feature_extraction import texture_feature_utils
        feat_path = default_paths.pyramid_texture_feat_path
        self.do_varpart=kwargs['do_varpart'] if 'do_varpart' in kwargs.keys() else True
        self.n_ori = kwargs['n_ori'] if 'n_ori' in kwargs.keys() else 4
        self.n_sf = kwargs['n_sf'] if 'n_sf' in kwargs.keys() else 4
        self.pca_type=kwargs['pca_type'] if 'pca_type' in kwargs.keys() else None
        
        self.group_all_hl_feats=kwargs['group_all_hl_feats'] if 'group_all_hl_feats' in kwargs.keys() else True
        
        if self.pca_type is not None:
            self.use_pca_feats=True
             # will use features where higher-level sub-sets have been reduced in dim with PCA.
            if self.pca_subject is not None:
                self.features_file = os.path.join(feat_path, 'PCA', \
                                              '%s_%dori_%dsf_%s_concat_wtsfromS%d_grid%d.h5py'%\
                                              (self.image_set, self.n_ori, self.n_sf, \
                                               self.pca_type, self.pca_subject, self.which_prf_grid))
                self.feature_column_labels, self.feature_type_names = \
                texture_feature_utils.get_feature_inds_pca('S%d'%self.pca_subject, self.pca_type, self.which_prf_grid)
                
            else:
                self.features_file = os.path.join(feat_path, 'PCA', \
                                              '%s_%dori_%dsf_%s_concat_grid%d.h5py'%\
                                              (self.image_set, self.n_ori, self.n_sf, \
                                               self.pca_type, self.which_prf_grid))
                self.feature_column_labels, self.feature_type_names = \
                    texture_feature_utils.get_feature_inds_pca(self.image_set, self.pca_type, self.which_prf_grid)
        else:
            self.use_pca_feats=False
            # will load from raw array (641 features).
            self.features_file = os.path.join(feat_path, \
                                              '%s_features_each_prf_%dori_%dsf_grid%d.h5py'%\
                                              (self.image_set, self.n_ori, self.n_sf, self.which_prf_grid))
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

        feat_path = default_paths.sketch_token_feat_path

        self.use_pca_feats = kwargs['use_pca_feats'] if 'use_pca_feats' in kwargs.keys() else False
        self.use_residual_st_feats = kwargs['use_residual_st_feats'] \
                                        if 'use_residual_st_feats' in kwargs.keys() else False
        self.use_grayscale_st_feats = kwargs['use_grayscale_st_feats'] \
                                        if 'use_grayscale_st_feats' in kwargs.keys() else False
        self.use_noavg = kwargs['use_noavg'] if 'use_noavg' in kwargs.keys() else False
        self.st_pooling_size = kwargs['st_pooling_size'] if 'st_pooling_size' in kwargs.keys() else 4
        self.st_use_avgpool = kwargs['st_use_avgpool'] if 'st_use_avgpool' in kwargs.keys() else False
        
        if self.use_noavg:
            self.use_pca_feats=True
            avg_str='_noavg'
            if self.st_use_avgpool:
                avg_str += '_avgpool'
            else:
                avg_str += '_maxpool'
            avg_str += '_poolsize%d'%self.st_pooling_size
        else:
            avg_str=''
                                                      
        if self.use_pca_feats:
            if self.use_grayscale_st_feats:
                if self.pca_subject is not None:
                    self.features_file = os.path.join(feat_path, 'PCA', \
                                                  '%s_grayscale%s_PCA_wtsfromS%d_grid%d.h5py'%\
                                                      (self.image_set, avg_str, self.pca_subject, self.which_prf_grid))
                else:
                    self.features_file = os.path.join(feat_path, 'PCA', \
                                                  '%s_grayscale%s_PCA_grid%d.h5py'%(self.image_set, avg_str, self.which_prf_grid))  
            else:
                if self.pca_subject is not None:
                    self.features_file = os.path.join(feat_path, 'PCA', \
                                                  '%s%s_PCA_wtsfromS%d_grid%d.h5py'%\
                                                      (self.image_set, avg_str, self.pca_subject, self.which_prf_grid))
                else:
                    self.features_file = os.path.join(feat_path, 'PCA', \
                                                  '%s%s_PCA_grid%d.h5py'%(self.image_set, avg_str, self.which_prf_grid))  
            with h5py.File(self.features_file, 'r') as file:
                feat_shape = np.shape(file['/features'])
                file.close()
            self.max_features = feat_shape[1]
        elif self.use_residual_st_feats:
            assert not self.use_grayscale_st_feats
            self.max_features = 150;
            self.features_file = os.path.join(feat_path, \
                          '%s_gabor_residuals_grid%d.h5py'%(self.image_set, self.which_prf_grid))
        elif self.use_grayscale_st_feats:
            self.max_features = 150;
            self.features_file = os.path.join(feat_path, \
                          '%s_features_grayscale_each_prf_grid%d.h5py'%(self.image_set, self.which_prf_grid))    
        else:
            self.max_features = 150;
            self.features_file = os.path.join(feat_path, \
                          '%s_features_each_prf_grid%d.h5py'%(self.image_set, self.which_prf_grid))
            
        self.do_varpart=False
        self.n_feature_types=1
    
    def __init_gist__(self, kwargs):
        
        assert(self.which_prf_grid==0)
        feat_path = default_paths.gist_feat_path
        self.n_ori = kwargs['n_ori'] if 'n_ori' in kwargs.keys() else 4
        self.n_blocks = kwargs['n_blocks'] if 'n_blocks' in kwargs.keys() else 4
        if self.n_ori==8 and self.n_blocks==4:
            self.max_features = 512;
            self.features_file = os.path.join(feat_path, '%s_gistdescriptors_%dori.h5py'%(self.image_set, self.n_ori))
        elif self.n_ori==4 and self.n_blocks==4:
            self.max_features = 256;
            self.features_file = os.path.join(feat_path, '%s_gistdescriptors_%dori.h5py'%(self.image_set, self.n_ori))
        elif self.n_ori==4 and self.n_blocks==2:
            self.features_file = os.path.join(feat_path, '%s_gistdescriptors_%dori_2blocks.h5py'%(self.image_set, self.n_ori))
            self.max_features = 64;
        self.do_varpart=False
        self.n_feature_types=1
        self.use_pca_feats=False
        
    def __init_color__(self, kwargs):
        
        self.use_noavg = kwargs['use_noavg'] if 'use_noavg' in kwargs.keys() else False
        if self.use_noavg:
            self.use_pca_feats=True
        else:
            self.use_pca_feats=False
            
        feat_path = default_paths.color_feat_path
        
        if self.use_pca_feats:
            if self.pca_subject is None:
                self.features_file = os.path.join(feat_path, 'PCA', \
                                              '%s_cielab_plus_sat_noavg_PCA_grid%d.h5py'%\
                                                  (self.image_set, self.which_prf_grid))
            else:
                self.features_file = os.path.join(feat_path, 'PCA', \
                               '%s_cielab_plus_sat_noavg_PCA_wtsfromS%d_grid%d.h5py'%\
                                (self.image_set, self.pca_subject, self.which_prf_grid))
            with h5py.File(self.features_file, 'r') as file:
                feat_shape = np.shape(file['/features'])
                file.close()
            self.max_features = feat_shape[1]
        else:
            self.use_pca_feats = False
            self.features_file = os.path.join(feat_path, \
                               '%s_cielab_plus_sat_grid%d.h5py'%(self.image_set, self.which_prf_grid))
            self.max_features=4
        
        self.do_varpart = False
        self.n_feature_types=1
        
    def __init_alexnet__(self,kwargs):
    
        from feature_extraction import extract_alexnet_features
        self.padding_mode = kwargs['padding_mode'] if 'padding_mode' in kwargs.keys() else 'reflect'
        if 'layer_name' not in kwargs.keys():
            raise ValueError('for alexnet, need to specify a layer name')
        self.layer_name = kwargs['layer_name']
        self.use_pca_feats = True
        self.blurface = kwargs['blurface'] if 'blurface' in kwargs.keys() else False
        self.use_noavg = kwargs['use_noavg'] if 'use_noavg' in kwargs.keys() else False
                                                      
        if self.blurface:
            feat_path = default_paths.alexnet_blurface_feat_path
        else:
            feat_path = default_paths.alexnet_feat_path
            
        alexnet_layer_names  = extract_alexnet_features.alexnet_layer_names
        n_features_each_layer = extract_alexnet_features.n_features_each_layer
     
        if self.use_noavg:
            avg_str='_noavg'
        else:
            avg_str=''
                                                      
        if self.pca_subject is not None:
            self.features_file = os.path.join(feat_path, 'PCA', \
              '%s_%s_reflect%s_PCA_wtsfromS%d_grid%d.h5py'%\
                  (self.image_set, self.layer_name, avg_str, self.pca_subject, self.which_prf_grid))
        else:
            self.features_file = os.path.join(feat_path, 'PCA', \
              '%s_%s_reflect%s_PCA_grid%d.h5py'%(self.image_set, self.layer_name, avg_str, self.which_prf_grid))
       
        layer_ind = [ll for ll in range(len(alexnet_layer_names)) \
                         if alexnet_layer_names[ll]==self.layer_name]
        assert(len(layer_ind)==1)
       
        with h5py.File(self.features_file, 'r') as file:
            feat_shape = np.shape(file['/features'])
            file.close()
        self.max_features = feat_shape[1]
        
        self.do_varpart=False
        self.n_feature_types=1
        
    def __init_resnet__(self, kwargs):
        
        from feature_extraction import extract_resnet_features
        self.model_architecture = kwargs['model_architecture'] if 'model_architecture' in kwargs.keys() else 'RN50'
        self.training_type = kwargs['training_type'] if 'training_type' in kwargs.keys() else 'clip'
        if 'layer_name' not in kwargs.keys():
            raise ValueError('need to specify a layer name')
        self.layer_name = kwargs['layer_name']
        self.use_noavg = kwargs['use_noavg'] if 'use_noavg' in kwargs.keys() else False
        
        self.use_pca_feats = True
        
        if self.training_type=='clip':
            feat_path = default_paths.clip_feat_path
            training_type_str=''
        elif self.training_type=='blurface':
            feat_path = default_paths.resnet50_blurface_feat_path
            training_type_str=''
        elif self.training_type=='imgnet':
            feat_path = default_paths.resnet50_feat_path
            training_type_str=''
        elif 'startingblurry' in self.training_type:
            feat_path = default_paths.resnet50_startingblurry_feat_path
            training_type_str='_%s'%self.training_type.split('startingblurry_')[1]
            
        if self.use_noavg:
            avg_str='_noavg'
        else:
            avg_str=''
          
        if self.pca_subject is not None:
            self.features_file = os.path.join(feat_path, 'PCA', \
              '%s_%s%s_%s%s_PCA_wtsfromS%d_grid%d.h5py'%(self.image_set, self.model_architecture, \
                                                         training_type_str,self.layer_name, avg_str, \
                                                         self.pca_subject, self.which_prf_grid))  
        else:
            self.features_file = os.path.join(feat_path, 'PCA', \
              '%s_%s%s_%s%s_PCA_grid%d.h5py'%(self.image_set, self.model_architecture, \
                                              training_type_str, self.layer_name, avg_str, \
                                              self.which_prf_grid))  
        
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