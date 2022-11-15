import numpy as np
import sys

"""
Feature loader class that can merge other smaller modules (useful for variance partitioning)
"""

class combined_feature_loader:
    
    def __init__(self, modules, module_names, do_varpart=False, include_solo_models=True, lambda_groups=None):
        
        super(combined_feature_loader, self).__init__()
        self.modules = modules
        self.module_names = module_names
        self.do_varpart = do_varpart
        self.include_solo_models = include_solo_models
        self.max_features = np.sum(np.array([module.max_features for module in self.modules]))
        
        self.n_prfs_each_mod = [module.n_prfs for module in self.modules]
        print('n_prfs_each_mod:')
        print(self.n_prfs_each_mod)
        if np.any(np.array(self.n_prfs_each_mod)!=self.n_prfs_each_mod[0]):
            # this is for when we're mixing modules that include pRFs with those that model full-image features
            assert(len(np.unique(self.n_prfs_each_mod))==2)
            self.n_prfs = np.max(self.n_prfs_each_mod)
            assert(np.min(self.n_prfs_each_mod)==1)
            self.mixing_n_prfs = True
            
        else:
            self.n_prfs = self.modules[0].n_prfs
            self.mixing_n_prfs = False
        print(self.n_prfs, self.mixing_n_prfs)
        self.lambda_groups = lambda_groups
        
    def clear_big_features(self):
        
        for module in self.modules:
            module.clear_big_features()
            
    def get_partial_versions(self):
        
        n_total_feat = self.max_features
        masks = np.ones((1,n_total_feat), dtype=int)
        names = ['full_combined_model']
            
        if self.do_varpart:

            # going to define "masks" that combine certain sub-sets of models features at a time
            # to be used for variance partitioning
            feature_start_ind = 0

            for mi, module in enumerate(self.modules):

                if self.include_solo_models:
                    # first a version that only includes the features in current module
                    new_mask = np.zeros((1, n_total_feat), dtype=int)
                    new_mask[0,feature_start_ind:feature_start_ind+module.max_features] = 1
                    masks = np.concatenate((masks, new_mask), axis=0)
                    names += ['just_' + self.module_names[mi]]

                if (len(self.modules)>2) or (not self.include_solo_models):        
                    # next a version that only everything but the features in current module
                    # (note if there are just 2 modules, this would be redundant)
                    new_mask = np.ones((1, n_total_feat), dtype=int)
                    new_mask[0,feature_start_ind:feature_start_ind+module.max_features] = 0
                    masks = np.concatenate((masks, new_mask), axis=0)
                    names += ['leave_out_' + self.module_names[mi]]

                # if the module has any subsets of features defined, will also do partial versions with those subsets only
                module_partial_masks, module_partial_names = module.get_partial_versions()
                if len(module_partial_names)>1:        
                    new_masks = np.zeros((len(module_partial_names)-1, n_total_feat), dtype=int)
                    new_masks[:,feature_start_ind:feature_start_ind+module.max_features] = module_partial_masks[1:]
                    masks = np.concatenate((masks, new_masks), axis=0)
                    names += [self.module_names[mi] + '_' + name + '_no_other_modules'for name in module_partial_names[1:]]

                    if len(module_partial_names)==3:        
                        # for this special case, also adding in some other combinations  
                        # if more than two subsets then this will get too complicated...
                        new_masks = np.ones((len(module_partial_names)-1, n_total_feat), dtype=int)
                        new_masks[:,feature_start_ind:feature_start_ind+module.max_features] = module_partial_masks[1:]
                        masks = np.concatenate((masks, new_masks), axis=0)
                        names += [self.module_names[mi] + '_' + name + '_plus_other_modules' for name in module_partial_names[1:]]

                feature_start_ind += module.max_features

        return masks, names
        
        
    def get_feature_group_inds(self):
        
        if self.lambda_groups is not None:
            assert(len(self.lambda_groups)==len(self.modules))
        print('lambda groups:')
        print(self.lambda_groups)
        sys.stdout.flush()
        print('in get_feature_group_inds')
        sys.stdout.flush()
        
        for mi, module in enumerate(self.modules):
            
            gi = module.get_feature_group_inds()
            gi = np.ones(np.shape(gi))
            if self.lambda_groups is None:
                gi *= mi
            else:
                gi *= self.lambda_groups[mi]
                
            if mi==0:
                group_inds = gi
            else:
                group_inds = np.concatenate([group_inds, gi], axis=0)
        
        print('done with get_feature_group_inds')
        sys.stdout.flush()
        
        return group_inds
    
    
    def load(self, images, prf_model_index, fitting_mode = True):

        for mi, module in enumerate(self.modules):
            
            if self.n_prfs_each_mod[mi]==1:
                # if this module is a full-image feature set, then always use prf index 0
                features, inds = module.load(images, 0)
            else:
                features, inds = module.load(images, prf_model_index)
            
            
            if mi==0:
                all_features_concat = features
                feature_inds_defined = inds
            else:
                all_features_concat = np.concatenate((all_features_concat, features), axis=1)
                feature_inds_defined = np.concatenate((feature_inds_defined, inds), axis=0)
        
        print('Final shape of concatenated features:')
        print(all_features_concat.shape)

        return all_features_concat, feature_inds_defined