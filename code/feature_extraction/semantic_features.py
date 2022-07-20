import numpy as np
import os
import pandas as pd

from utils import default_paths, nsd_utils
from model_fitting import initialize_fitting

class semantic_feature_loader:
    
    def __init__(self, subject, which_prf_grid, feature_set, **kwargs):
    
        self.subject = subject
        self.feature_set = feature_set  
        self.which_prf_grid = which_prf_grid
        self.n_prfs = initialize_fitting.get_prf_models(which_grid=self.which_prf_grid).shape[0]
        self.remove_missing = kwargs['remove_missing'] if 'remove_missing' in kwargs.keys() else False
        self.use_pca_feats = kwargs['use_pca_feats'] if 'use_pca_feats' in kwargs.keys() else False
        
        self.__get_categ_exclude__()
      
        if self.feature_set=='indoor_outdoor':
            self.features_file = os.path.join(default_paths.stim_labels_root, \
                                          'S%d_indoor_outdoor.csv'%self.subject)
            self.same_labels_all_prfs=True
            self.n_features = 2
            
        elif self.use_pca_feats:
            self.labels_folder = os.path.join(default_paths.stim_labels_root, \
                                         'S%d_within_prf_grid%d_PCA'%(self.subject, self.which_prf_grid))
            self.same_labels_all_prfs=False
            self.features_file = os.path.join(self.labels_folder, \
                          'S%d_%s_prf0_PCA.csv'%(self.subject, self.feature_set))
            if self.feature_set=='coco_things_categ':
                self.n_features = 80
            elif self.feature_set=='coco_stuff_categ':
                self.n_features = 92   
                
        else:

            self.labels_folder = os.path.join(default_paths.stim_labels_root, \
                                         'S%d_within_prf_grid%d'%(self.subject, self.which_prf_grid))
            self.same_labels_all_prfs=False
            if self.feature_set=='natural_humanmade':            
                self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_natural_humanmade_prf0.csv'%(self.subject))          
                self.n_features = 2
            elif self.feature_set=='real_world_size':            
                self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_realworldsize_prf0.csv'%(self.subject))          
                self.n_features = 3
            elif 'coco_things' in self.feature_set:
                self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_cocolabs_binary_prf0.csv'%(self.subject))
                
                if 'supcateg' in self.feature_set:                    
                    self.n_features = 12
                elif 'material_diagnostic' in self.feature_set:
                    # will select just a sub-set of the coco things categories, based on
                    # whether the objects have material/color as a diagnostic feature or not
                    assert(not self.remove_missing and not self.use_pca_feats)
                    groups_fn = os.path.join(default_paths.stim_labels_root, \
                                                   'Material_diagnostic_categ.npy')
                    material_groups = np.load(groups_fn, allow_pickle=True).item()                    
                    if 'not_material_diagnostic' in self.feature_set:
                        self.categ_names_use = [kk for kk in material_groups.keys() \
                                             if material_groups[kk]==0]
                    else:
                        self.categ_names_use = [kk for kk in material_groups.keys() \
                                             if material_groups[kk]==1]
                    self.n_features = len(self.categ_names_use)                    
                else:
                    # use all the basic-level coco things categories
                    self.n_features = 80
                    
            elif 'coco_stuff' in self.feature_set:
                self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_cocolabs_stuff_binary_prf0.csv'%(self.subject))
                if 'supcateg' in self.feature_set:                    
                    self.n_features = 16
                else:
                    self.n_features = 92           
            else:
                self.features_file = os.path.join(self.labels_folder, \
                                      'S%d_cocolabs_binary_prf0.csv'%(self.subject))
                self.n_features = 2
    
        self.max_features = self.n_features
        
        if not os.path.exists(self.features_file):
            raise RuntimeError('Looking at %s for precomputed features, not found.'%self.features_file)

        self.features_in_prf = None
        
    def __get_categ_exclude__(self):
        
        if self.remove_missing:
            # find any features that are missing in the training set for any subject, in any pRF.
            # will choose to exclude these columns from fitting for all subjects.
            labels_folder = os.path.join(default_paths.stim_labels_root)
            fn2load = os.path.join(labels_folder, \
                           'Coco_label_counts_all_prf_grid%d.npy'%self.which_prf_grid)   
            counts = np.load(fn2load, allow_pickle=True).item()
            things_counts_trn = counts['things_cat_counts_trntrials']
            self.things_inds_exclude = np.any(np.any(things_counts_trn==0, axis=0), axis=0)
            stuff_counts_trn = counts['stuff_cat_counts_trntrials']
            self.stuff_inds_exclude = np.any(np.any(stuff_counts_trn==0, axis=0), axis=0)
            print('excluding %d things categories and %d stuff categories (not enough instances)'\
                 %(np.sum(self.things_inds_exclude),np.sum(self.stuff_inds_exclude)))
        else:
            self.things_inds_exclude = None
            self.stuff_inds_exclude = None
            
    def clear_big_features(self):
        
        print('Clearing features from memory')
        self.features_in_prf = None 

    def get_partial_versions(self):

        partial_version_names = ['full_model']
        masks = np.ones([1,self.max_features])

        return masks, partial_version_names

    def __load_precomputed_features__(self, image_inds, prf_model_index):

        if self.same_labels_all_prfs:
            print('Loading pre-computed features from %s'%self.features_file)        
            coco_df = pd.read_csv(self.features_file, index_col=0)
            labels = np.array(coco_df)
            colnames = list(coco_df.keys())  
            
        elif self.use_pca_feats:            
            self.features_file = os.path.join(self.labels_folder, \
                          'S%d_%s_prf%d_PCA.csv'%(self.subject, self.feature_set, prf_model_index))
            print('Loading pre-computed features from %s'%self.features_file)
            pca_df = pd.read_csv(self.features_file, index_col=0, dtype=np.float32)
            labels = np.array(pca_df)
            colnames = ['pc%d'%pc for pc in range(labels.shape[1])]
            
        else:
            if self.feature_set=='natural_humanmade':
                self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_natural_humanmade_prf%d.csv'%(self.subject, prf_model_index))
                print('Loading pre-computed features from %s'%self.features_file)
                nat_hum_df = pd.read_csv(self.features_file, index_col=0)                
                labels = np.array(nat_hum_df)
                colnames = list(nat_hum_df.keys())
            elif self.feature_set=='real_world_size':
                self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_realworldsize_prf%d.csv'%(self.subject, prf_model_index))
                print('Loading pre-computed features from %s'%self.features_file)
                size_df = pd.read_csv(self.features_file, index_col=0)                
                # use small, medium, large.
                labels = np.array(size_df).astype(np.float32)
                colnames = list(size_df.keys())
            elif 'coco_stuff' in self.feature_set:
                self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_cocolabs_stuff_binary_prf%d.csv'%(self.subject, \
                                                                         prf_model_index))
                print('Loading pre-computed features from %s'%self.features_file)
                coco_df = pd.read_csv(self.features_file, index_col=0)
                if 'supcateg' in self.feature_set:
                    labels = np.array(coco_df)[:,0:16]
                    colnames = list(coco_df.keys())[0:16]
                else:
                    labels = np.array(coco_df)[:,16:]   
                    colnames = list(coco_df.keys())[16:]
            else:
                self.features_file = os.path.join(self.labels_folder, \
                                  'S%d_cocolabs_binary_prf%d.csv'%(self.subject, prf_model_index))
                print('Loading pre-computed features from %s'%self.features_file)
                coco_df = pd.read_csv(self.features_file, index_col=0)
                if 'supcateg' in self.feature_set:
                    labels = np.array(coco_df)[:,0:12]
                    colnames = list(coco_df.keys())[0:12]
                elif 'categ' in self.feature_set:
                    labels = np.array(coco_df)[:,12:92]   
                    colnames = list(coco_df.keys())[12:92]
                    if 'material_diagnostic' in self.feature_set:
                        colnames = [cc.split('.1')[0] for cc in colnames]
                        columns_use = np.isin(colnames, self.categ_names_use)
                        assert(np.sum(columns_use)==self.n_features)
                        labels = labels[:,columns_use]
                        colnames = np.array(colnames)[columns_use]
                   
                elif self.feature_set=='animacy':    
                    supcat_labels = np.array(coco_df)[:,0:12]
                    animate_supcats = [1,9]
                    inanimate_supcats = [ii for ii in range(12)\
                                         if ii not in animate_supcats]
                    has_animate = np.any(np.array([supcat_labels[:,ii]==1 \
                                                   for ii in animate_supcats]), axis=0)
                    has_inanimate = np.any(np.array([supcat_labels[:,ii]==1 \
                                                for ii in inanimate_supcats]), axis=0)
                    labels = np.concatenate([has_animate[:,np.newaxis], \
                                             has_inanimate[:,np.newaxis]], axis=1)
                    colnames = ['has_animate','has_inanimate']
                 
                else:
                    has_label = np.any(np.array(coco_df)[:,0:12]==1, axis=1)
                    label1 = np.array(coco_df[self.feature_set])[:,np.newaxis]
                    label2 = (label1==0) & (has_label[:,np.newaxis])
                    labels = np.concatenate([label1, label2], axis=1)
                    colnames = ['has_%s'%self.feature_set, 'has_other']
                    
        print('using feature set: %s'%self.feature_set)        
        print(colnames)

        if self.use_pca_feats:
            self.is_defined_in_prf = np.zeros((self.max_features,),dtype=bool)
            self.is_defined_in_prf[0:labels.shape[1]] = 1;
        elif self.remove_missing:
            assert(labels.shape[1]==self.max_features)
            if self.feature_set=='coco_things_categ':
                labels = labels[:,~self.things_inds_exclude]
                missing = self.things_inds_exclude
            elif self.feature_set=='coco_stuff_categ':
                labels = labels[:,~self.stuff_inds_exclude]
                missing = self.stuff_inds_exclude
            self.is_defined_in_prf = ~missing  
        else:
            assert(labels.shape[1]==self.max_features)
            self.is_defined_in_prf = np.ones((self.max_features,),dtype=bool)

        labels = labels[image_inds,:].astype(np.float32)           
        self.features_in_prf = labels;
        
        # print counts to verify things are working ok
        if self.max_features==2 and labels.shape[1]==2:
            print('num 1/1, 1/0, 0/1, 0/0:')
            print([np.sum((labels[:,0]==1) & (labels[:,1]==1)), \
                   np.sum((labels[:,0]==1) & (labels[:,1]==0)),\
                   np.sum((labels[:,0]==0) & (labels[:,1]==1)),\
                   np.sum((labels[:,0]==0) & (labels[:,1]==0))])
        else:
            print('sum each column:')
            print(np.sum(labels, axis=0))
            
        print('Size of features array for this image set is:')
        print(self.features_in_prf.shape)
        
    
    def load(self, image_inds, prf_model_index):
    
        if (not self.same_labels_all_prfs) or (self.features_in_prf is None):
            self.__load_precomputed_features__(image_inds, prf_model_index)
        
        features = self.features_in_prf
        feature_inds_defined = self.is_defined_in_prf
        
        assert(len(feature_inds_defined)==self.max_features)
        assert(np.sum(feature_inds_defined)==features.shape[1])        
        assert(features.shape[0]==len(image_inds))
        print('Final size of feature matrix is:')
        print(features.shape)
          
        return features, feature_inds_defined
     
    