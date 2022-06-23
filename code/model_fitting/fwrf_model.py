import sys
import time
import copy
import numpy as np
import gc

import torch

from utils import torch_utils, stats_utils

"""
General code for fitting a 'feature weighted receptive field' model to fmri data - looping over many candidate pRF 
models for each voxel, find a set of weights that best predict its responses based on feature space of interest.
Can work for many different types of feature spaces (visual, semantic)
Requires a "feature_loader" object, from feature_extraction/fwrf_features.py
"""

class encoding_model():
    
    def __init__(self, \
                 feature_loader,
                 **kwargs):
        
        self.feature_loader = feature_loader
        
        default_lambdas = np.logspace(np.log(0.01),np.log(10**5+0.01),9, \
                                      dtype=np.float32, base=np.e) - 0.01
        self.lambdas = kwargs['lambdas'] if 'lambdas' in kwargs.keys() else default_lambdas
        
        self.best_model_each_voxel = kwargs['best_model_each_voxel'] \
            if 'best_model_each_voxel' in kwargs.keys() else None
        # are we fitting prfs now, or were they done before?
        self.fitting_prfs_now = self.best_model_each_voxel is None
        
        self.zscore = kwargs['zscore'] if 'zscore' in kwargs.keys() else True
        self.add_bias = kwargs['add_bias'] if 'add_bias' in kwargs.keys() else True
        self.voxel_batch_size = kwargs['voxel_batch_size'] \
            if 'voxel_batch_size' in kwargs.keys() else 100
        self.sample_batch_size = kwargs['sample_batch_size'] \
            if 'voxel_batch_size' in kwargs.keys() else 100
        self.prfs_fit_mask = kwargs['prfs_fit_mask'] \
            if 'prfs_fit_mask' in kwargs.keys() else None
        
        self.shuffle_data = kwargs['shuffle_data'] \
            if 'shuffle_data' in kwargs.keys() else False
        self.n_shuff_iters = kwargs['n_shuff_iters'] \
            if 'n_shuff_iters' in kwargs.keys() else None
        self.shuff_rnd_seed = kwargs['shuff_rnd_seed'] \
            if 'shuff_rnd_seed' in kwargs.keys() else None
        self.shuff_batch_size = kwargs['shuff_batch_size'] \
            if 'shuff_batch_size' in kwargs.keys() else None
        n_shuff_batches = int(np.ceil(self.n_shuff_iters/self.shuff_batch_size))
        self.shuff_batch_inds = [np.arange(bb*self.shuff_batch_size, \
                                       np.min([(bb+1)*self.shuff_batch_size, self.n_shuff_iters])) \
                                       for bb in range(n_shuff_batches)]
            
        if self.shuffle_data:
            # we only do the permutation analysis once pRFs are fit
            assert(not self.fitting_prfs_now)
            
        self.device = kwargs['device'] if 'device' in kwargs.keys() else torch.device('cpu:0')
        self.dtype = kwargs['dtype'] if 'dtype' in kwargs.keys() else np.float32
        self.debug = kwargs['debug'] if 'debug' in kwargs.keys() else False

        print ('dtype = %s' % self.dtype)
        print ('device = %s' % self.device)
        
        # clear any stored features from feature loader's memory    
        self.feature_loader.clear_big_features()
        
        # max features for any pRF - can be different actual number for diff pRFs (due to PCA)    
        self.max_features = self.feature_loader.max_features 
        
        # Decide whether to do any "partial" versions of the models (leaving out subsets of features)
        # Purpose is for variance partition
        self.masks, self.partial_version_names = self.feature_loader.get_partial_versions()
        self.n_partial_versions = len(self.partial_version_names) # will be one if skipping varpart
    
        if self.add_bias:
            # including intercept 
            self.masks = np.concatenate([self.masks, \
                             np.ones([self.masks.shape[0],1])], axis=1) 
        self.masks = np.transpose(self.masks)
        # masks is [n_features_total (including intercept) x n_partial_versions]
        
        self.n_prfs = self.feature_loader.n_prfs
        if self.prfs_fit_mask is not None:
            assert(len(self.prfs_fit_mask)==self.n_prfs)
            # only use this in from-scratch pRF fitting
            assert(self.fitting_prfs_now)
            
            
    
    def fit(self, \
            image_inds_trn, \
            voxel_data_trn, \
            image_inds_holdout, \
            voxel_data_holdout, \
            **kwargs, \
           ):
        
            
        self.trials_use_each_prf_trn = kwargs['trials_use_each_prf_trn'] \
            if 'trials_use_each_prf_trn' in kwargs.keys() else None
        self.trials_use_each_prf_holdout = kwargs['trials_use_each_prf_holdout'] \
            if 'trials_use_each_prf_holdout' in kwargs.keys() else None
        
        self.__init_for_fit__(image_inds_trn,voxel_data_trn,\
                              image_inds_holdout,voxel_data_holdout)
        
        # clear any stored features from feature loader's memory    
        self.feature_loader.clear_big_features()
        
        start_time = time.time()
        vox_loop_time = 0

        print ('---------------------------------------\n')
        
        with torch.no_grad(): # make sure local gradients are off to save memory
                
            # Looping over pRFs here
            for mm in range(self.n_prfs):
                
                if self.debug and mm>1:
                    # this is just a way of testing code, stop after prf 1
                    break
                    
                if self.prfs_fit_mask is not None and (not self.prfs_fit_mask[mm]):
                    # this is for skipping a pRF and moving to next one
                    print('skipping pRF %d, based on prfs_fit_mask'%mm)
                    continue
                    
                print('\nProcessing prf %d of %d'%(mm, self.n_prfs))

                self.__fit_one_prf__(mm)
                
                gc.collect()
        
                sys.stdout.flush()
                
        # Finish up, prepare to return items of interest
        self.best_weights = self.best_w_params[:,0:self.max_features,:]  
        if self.add_bias:
            self.best_biases = self.best_w_params[:,-1,:]       
        else: 
            self.best_biases = None
        
        # trying to clear some space here...
        self.best_w_params = None
        self.voxel_data_trn = None
        self.image_inds_trn = None
        self.voxel_data_holdout = None
        self.image_inds_holdout = None
        self.shuff_inds_trn = None
        self.shuff_inds_out = None
        self.image_inds_concat = None
        self.trials_use_each_prf_trn = None
        self.trials_use_each_prf_holdout = None
        gc.collect()
        
        # This step clears any loaded arrays out of feature loader (no longer needed)
        self.feature_loader.clear_big_features()
        gc.collect()
        
        sys.stdout.flush()
       
    
    def __init_for_fit__(self, \
                        image_inds_trn, \
                        voxel_data_trn, \
                        image_inds_holdout, \
                        voxel_data_holdout):
                
        self.voxel_data_trn = voxel_data_trn;
        self.voxel_data_holdout = voxel_data_holdout;
        
        # concatenate trn/val inds here, makes it easier to extract features
        self.image_inds_concat = np.concatenate([image_inds_trn, image_inds_holdout], axis=0)
        n_trials = len(self.image_inds_concat)
        # make indices for how to get train/val trials back out of concat trials
        self.trninds = np.arange(n_trials)<len(image_inds_trn)
        self.outinds = np.arange(n_trials)>=len(image_inds_trn)

        self.n_voxels = self.voxel_data_trn.shape[1]   

        # Initialize arrays to store model fitting params
        n_params = self.max_features
        if self.add_bias:
            n_params += 1
        
        if self.shuffle_data:
            self.best_w_params = np.zeros(shape=(self.n_voxels, \
                                             n_params, \
                                             self.n_partial_versions, \
                                             self.n_shuff_iters), dtype=self.dtype)
        else:
            self.best_w_params = np.zeros(shape=(self.n_voxels, \
                                                 n_params, \
                                                 self.n_partial_versions), dtype=self.dtype)
            
        self.best_prf_models = np.full(shape=(self.n_voxels, \
                                             self.n_partial_versions), fill_value=-1, dtype=int)   
        self.best_lambdas = np.full(shape=(self.n_voxels, \
                                             self.n_partial_versions), fill_value=-1, dtype=int)
        self.best_losses = np.full(fill_value=np.inf, \
                                   shape=(self.n_voxels, \
                                          self.n_partial_versions), dtype=self.dtype)

        # params needed if z-scoring
        if self.zscore:        
            self.features_mean = np.zeros(shape=(self.n_prfs, self.max_features), dtype=self.dtype)
            self.features_std  = np.zeros(shape=(self.n_prfs, self.max_features), dtype=self.dtype)
            print('will z-score each column')
        else:
            self.features_mean = None
            self.features_std = None
            print('will not z-score')

        if self.shuffle_data:
            # prep for permutation test, if doing
            if self.trials_use_each_prf_trn is not None:
                raise ValueError('cannot specify a trial subset when also doing permutation test')
            if self.shuff_rnd_seed is not None:
                np.random.seed(self.shuff_rnd_seed)
            n_trn = len(image_inds_trn)
            n_out = len(image_inds_holdout)
            self.shuff_inds_trn = np.zeros((n_trn,self.n_shuff_iters),dtype=int)
            self.shuff_inds_out = np.zeros((n_out,self.n_shuff_iters),dtype=int)
            for xx in range(self.n_shuff_iters):
                self.shuff_inds_trn[:,xx] = np.random.permutation(n_trn)
                self.shuff_inds_out[:,xx] = np.random.permutation(n_out)
            
    def __fit_one_prf__(self, mm):

        # Initialize some variables for this pRF
        
        if self.fitting_prfs_now:
            # if we're currently fitting pRFs: then need to keep track of whether 
            # current prf is better than running best, for each voxel.
            # Note that for a given voxel, the same pRF is always used for full and partial models.
            # So this variable only tracks which pRFs give improvement for FULL model.
            full_model_improved = np.zeros((self.n_voxels,),dtype=bool)
            voxels_to_fit = np.arange(0, self.n_voxels)
        else:
            # this is the case where we already know pRFs, so we're only going to fit those 
            # voxels whose best pRF was the current one.
            voxels_to_fit = np.where(self.best_model_each_voxel==mm)[0]
            full_model_improved = None
            
        if len(voxels_to_fit)==0:
            print('No voxels have this pRF saved as their best model, skipping it.')
            return
                   
        n_voxel_batches = int(np.ceil(len(voxels_to_fit)/self.voxel_batch_size))

        # Load features for the desired pRF, across all images (both train/holdout)       
        features, feature_inds_defined = self.feature_loader.load(self.image_inds_concat, \
                                                                  prf_model_index = mm, \
                                                                  fitting_mode=True)
        # Features is size [n_trials x n_features_actual]
        # n_features_actual may be less than self.max_features; 
        # because max_features is the largest number possible for any pRF.
        # feature_inds_defined is a boolean of length max_features, and indicates which of 
        # the features in max_features are included in features_actual.        
        n_features_actual = features.shape[1]

        if self.zscore:                
            features_m = np.mean(features, axis=0, keepdims=True) #[:trn_size]
            features_s = np.std(features, axis=0, keepdims=True) + 1e-6          
            features -= features_m
            features /= features_s   
            # saving these for later so we can exactly reproduce this normalization
            # when doing validation pass...
            self.features_mean[mm,feature_inds_defined] = features_m
            self.features_std[mm,feature_inds_defined] = features_s
        if self.add_bias:
            features = np.concatenate([features, np.ones(shape=(len(features), 1), dtype=self.dtype)], axis=1)
            feature_inds_defined = np.concatenate((feature_inds_defined, [True]), axis=0)

        # now separate out the train/holdout images
        trn_features = features[self.trninds,:]
        out_features = features[self.outinds,:]

        # can further specify a trial subset here
        if self.trials_use_each_prf_trn is not None:
            # select a subset of trials to work with, on per-pRF basis
            trials_use_trn = self.trials_use_each_prf_trn[:,mm]
            trials_use_holdout = self.trials_use_each_prf_holdout[:,mm]
            trn_features = trn_features[trials_use_trn,:]
            trn_data_use = self.voxel_data_trn[trials_use_trn,:]
            out_features = out_features[trials_use_holdout,:]
            out_data_use = self.voxel_data_holdout[trials_use_holdout,:]
        else:
            trn_data_use = self.voxel_data_trn
            out_data_use = self.voxel_data_holdout

        # check how many trials we're left with now...
        if trn_data_use.shape[0]==0 or out_data_use.shape[0]==0:
            # if insufficient trials to work with this pRF - skip it.
            # this only happens when the trial counts are subsampled in various control analyses.
            assert(self.best_model_each_voxel is not None) # this can never happen when we are actually fitting pRFs.
            print('prf %d - not enough trials to fit. skipping voxels with this pRF!'%(mm))
            return
        print('prf %d - using %d training trials and %d held-out trials'\
                  %(mm, trn_data_use.shape[0], out_data_use.shape[0]))

        # Looping over versions of the model w different features set to zero (variance partition)
        for pp in range(self.n_partial_versions):

            print('\nFitting version %d of %d: %s, '%(pp, self.n_partial_versions, self.partial_version_names[pp]))

            # nonzero_inds_full is length max_features (or max_features+1 if bias=True)
            # same size as the final params matrices will be.
            nonzero_inds_full = np.logical_and(self.masks[:,pp], feature_inds_defined)             
            # nonzero_inds_full is restricted to just indices that are defined for this prf 
            # (same size as features)
            nonzero_inds_short = self.masks[feature_inds_defined,pp]==1

            # Send matrices to gpu    
            _xtrn = torch_utils._to_torch(trn_features[:, nonzero_inds_short], device=self.device)
            _xout = torch_utils._to_torch(out_features[:, nonzero_inds_short], device=self.device)   

            # Do part of the matrix math involved in ridge regression optimization out of the loop, 
            # because this part will be same for all the voxels.
            _cof = self.__cofactor_fn_cpu__(_xtrn) 

             # Now looping over batches of voxels (only reason is because can't store all in memory at same time)
            for vv in range(n_voxel_batches):
                
                vinds = np.arange(self.voxel_batch_size*vv, \
                          np.min([self.voxel_batch_size*(vv+1), len(voxels_to_fit)]))
                voxel_batch_inds = voxels_to_fit[vinds]
        
                if self.shuffle_data:                    
                    self.__fit_voxel_batch_shuffle__(_cof, _xout, \
                                trn_data_use, out_data_use, \
                                nonzero_inds_full, \
                                full_model_improved, voxels_to_fit, \
                                mm, pp, voxel_batch_inds)              
                else:                       
                    self.__fit_voxel_batch__(_cof, _xout, \
                                trn_data_use, out_data_use, \
                                nonzero_inds_full, \
                                full_model_improved, voxels_to_fit, \
                                mm, pp, voxel_batch_inds)
                    
                gc.collect()
                       
    def __fit_voxel_batch__(self, _cof, _xout, \
                                trn_data_use, out_data_use, \
                                nonzero_inds_full, \
                                full_model_improved, voxels_to_fit, \
                                mm, pp, voxel_batch_inds):
                
                 
        
        
        # Send matrices to gpu
        _vtrn = torch_utils._to_torch(trn_data_use[:,voxel_batch_inds], device=self.device)
        _vout = torch_utils._to_torch(out_data_use[:,voxel_batch_inds], device=self.device)

        # Fit weights and get prediction loss here
        _beta, _loss = self.__loss_fn__(_cof, _vtrn, _xout, _vout) 

        # choose best lambda value and the loss that went with it.
        # loss is size: [lambdas x voxels]
        _best_loss_values, _best_lambda_index = torch.min(_loss, dim=0)

        # back to numpy now
        best_loss_values = torch_utils.get_value(_best_loss_values)
        best_lambda_index =  torch_utils.get_value(_best_lambda_index)
        
        # betas is size: [lambdas x features x voxels]
        betas = torch_utils.get_value(_beta)
        # choose betas that go with best lambda (reduce to size [features x voxels])
        betas = np.array([betas[best_lambda_index[ii],:,ii] for ii in range(len(best_lambda_index))]).T
           
            
        # decide what to do next...
        if self.fitting_prfs_now:

            if pp==0:
                # here is where we decide what pRF is best for this voxel.
                # Did the current pRF give lower loss than others?
                improved_voxels = best_loss_values < self.best_losses[voxel_batch_inds,0]
                full_model_improved[voxel_batch_inds] = improved_voxels
            else:
                # if we're working with a "partial" model right now, then we'll always defer to 
                # the pRF that was best for the "full" model.
                improved_voxels = full_model_improved[voxel_batch_inds]                    
        else:

            # if this is the pre-computed pRFs situation, then we'll save
            # betas for all the voxels here regardless of actual loss.
            improved_voxels = np.ones((len(voxel_batch_inds),),dtype=bool)

        # finally, saving parameters for this pRF fit
        if np.sum(improved_voxels)>0:

            # Create indices for the improved voxels into my full array over all voxels
            voxel_inds_save = voxel_batch_inds[improved_voxels]

            self.best_prf_models[voxel_inds_save,pp] = mm

            self.best_lambdas[voxel_inds_save,pp] = best_lambda_index[improved_voxels]
            self.best_losses[voxel_inds_save,pp] = best_loss_values[improved_voxels]                     

            # taking the weights associated with the best lambda value
            # if there are fewer features defined than self.max_features,
            # then we'll have some zeros in this matrix 
            best_w_tmp = self.best_w_params[voxel_inds_save,:,pp]
            # best_w_tmp = copy.deepcopy(self.best_w_params[voxel_inds_save,:,pp])
            best_w_tmp[:,nonzero_inds_full] = betas[:,improved_voxels].T                   
            best_w_tmp[:,~nonzero_inds_full] = 0.0 # make sure to fill zeros here

            # put this back into full sized array.
            self.best_w_params[voxel_inds_save,:,pp] = best_w_tmp
            
        best_w_tmp = None
        betas_all = None
        best_lambda_index = None
        best_loss_values = None
        gc.collect()
            
    def __fit_voxel_batch_shuffle__(self, _cof, _xout, \
                                    trn_data_use, out_data_use, \
                                    nonzero_inds_full, \
                                    full_model_improved, voxels_to_fit, \
                                    mm, pp, voxel_batch_inds):
                
        # Send matrices to gpu
        trnbatch = trn_data_use[:,voxel_batch_inds]
        outbatch = trn_data_use[:,voxel_batch_inds]

        betas_all = np.zeros((len(voxel_batch_inds), _xout.shape[1], self.n_shuff_iters), dtype=self.dtype)
        
        # do shuffled fitting in batches to prevent memory overload
        for bb, batch_inds in enumerate(self.shuff_batch_inds):
            
            print('permutation test, batch %d of %d'%(bb, len(self.shuff_batch_inds)))
            
            _vtrn = torch_utils._to_torch(trn_data_use[:,voxel_batch_inds], device=self.device)
            _vout = torch_utils._to_torch(out_data_use[:,voxel_batch_inds], device=self.device)
        
            # create trial-shuffled datasets here, multiple iterations of shuffling
            # don't make a new variable name here, trying to save memory
            _vtrn = torch.cat([_vtrn[self.shuff_inds_trn[:,ii],:,None] for ii in batch_inds], axis=2)
            _vout = torch.cat([_vout[self.shuff_inds_out[:,ii],:,None] for ii in batch_inds], axis=2)
            # size [n_trials x n_voxels x n_shuff_iters]
           
            # Fit weights and get prediction loss here
            _beta, _loss = self.__loss_fn__(_cof, _vtrn, _xout, _vout) 
            # betas size [lambdas x features x voxels x shuff_iters]

            # choose best lambda value and the loss that went with it.
            # loss is size: [lambdas x voxels x n_shuff_iters]
            _best_loss_values, _best_lambda_index = torch.min(_loss, dim=0)
            # each size [voxels x shuff_iters]

            # back to numpy now
            best_loss_values = torch_utils.get_value(_best_loss_values)
            best_lambda_index =  torch_utils.get_value(_best_lambda_index)

            # betas is size: [lambdas x features x voxels x n_shuff_iters]
            betas = torch_utils.get_value(_beta)
            # choose betas that go with best lambda 
            betas = np.array([[betas[best_lambda_index[ii,jj],:,ii,jj] \
                               for ii in range(len(best_lambda_index))] \
                               for jj in range(len(batch_inds))])
            # will be size [voxels x features x n_shuff_iters]
            betas = np.moveaxis(betas,[0,1,2],[2,0,1])

            print('size of best betas:')
            print(betas.shape)
            
            betas_all[:,:,batch_inds] = betas
            
            betas = None
            gc.collect()
            
        print('size of betas_all:')
        print(betas_all.shape)
            
        # for permutation analysis, we already fit pRFs so always saving all voxels.
        voxel_inds_save = voxel_batch_inds

        self.best_prf_models[voxel_inds_save,pp] = mm

        # only saving one lambda and loss, because they'll get very large and we don't really need them.
        self.best_lambdas[voxel_inds_save,pp] = best_lambda_index[:,0]
        self.best_losses[voxel_inds_save,pp] = best_loss_values[:,0]                     

        # make sure to save all the weights, because we still need to evaluate the model
        # taking the weights associated with the best lambda value
        # if there are fewer features defined than self.max_features,
        # then we'll have some zeros in this matrix 
        best_w_tmp = self.best_w_params[voxel_inds_save,:,pp,:] # don't copy, save space
        # best_w_tmp = copy.deepcopy(self.best_w_params[voxel_inds_save,:,pp,:])
        best_w_tmp[:,nonzero_inds_full,:] = betas_all             
        best_w_tmp[:,~nonzero_inds_full,:] = 0.0 # make sure to fill zeros here

        # put this back into full sized array.
        self.best_w_params[voxel_inds_save,:,pp,:] = best_w_tmp

        best_w_tmp = None
        betas_all = None
        best_lambda_index = None
        best_loss_values = None
        gc.collect()
                
                
    def __cofactor_fn_cpu__(self, _x):
        '''
        Generating a matrix needed to solve ridge regression model for each lambda value.
        Ridge regression (Tikhonov) solution is :
        w = (X^T*X + I*lambda)^-1 * X^T * Y
        This func will return (X^T*X + I*lambda)^-1 * X^T. 
        So once we have that, can just multiply by training data (Y) to get weights.
        returned size is [nLambdas x nFeatures x nTrials]
        This version makes sure that the torch inverse operation is done on the cpu, and in 
        floating point-64 precision. 
        Otherwise it gives a numerically different result than numpy operations.

        '''
        device_orig = _x.device
        type_orig = _x.dtype
        # switch to this specific format which works with inverse
        _x = _x.to('cpu').to(torch.float64)

        try: 
            _f = torch.stack([(torch.mm(torch.t(_x), _x) + \
                               torch.eye(_x.size()[1], device='cpu', dtype=torch.float64) * l).inverse() \
                              for l in self.lambdas], axis=0) 

        except RuntimeError:
            # problem with inverse - print some info to help diagnose the problem.
            # usually due to zero columns or duplicate columns.
            print('WARNING: Problem with inverse in _cofactor_fn_cpu.')
            print('Size of _x (trials x features):')
            print(_x.shape)
            print('Rank of _x:')
            print(torch.matrix_rank(_x))
            # to prevent a crash, replace 0 with a small lambda value, just temporarily
            lambdas_adjusted = copy.deepcopy(self.lambdas)
            lambdas_adjusted[lambdas_adjusted==0] = 10e-9
            print('Trying again with these lambda values:')
            print(lambdas_adjusted)
            _f = torch.stack([(torch.mm(torch.t(_x), _x) + \
                               torch.eye(_x.size()[1], device='cpu', dtype=torch.float64) * l).inverse() \
                              for l in lambdas_adjusted], axis=0) 

        # [#lambdas, #feature, #feature] 
        cof = torch.tensordot(_f, _x, dims=[[2],[1]]) # [#lambdas, #feature, #sample]

        # put back to whatever way it was before, so that we can continue with other operations as usual
        return cof.to(device_orig).to(type_orig)



    def __loss_fn__(self, _cofactor, _vtrn, _xout, _vout):
        '''
        Calculate loss given "cofactor" from cofactor_fn, training data, held-out design matrix, held out data.
        returns weights (betas) based on equation
        w = (X^T*X + I*lambda)^-1 * X^T * Y
        also returns loss for these weights w the held out data. SSE is loss func here.
        '''

        _beta = torch.tensordot(_cofactor, _vtrn, dims=[[2], [0]]) # [#lambdas, #feature, #voxel]
        _pred = torch.tensordot(_xout, _beta, dims=[[1],[1]]) # [#samples, #lambdas, #voxels]
        _loss = torch.sum(torch.pow(_vout[:,None,:] - _pred, 2), dim=0) # [#lambdas, #voxels]
        
             
        return _beta, _loss


    def validate(self, \
                 voxel_data_val, \
                 image_inds_val, \
                 **kwargs):
        
        self.trials_use_each_prf_val = kwargs['trials_use_each_prf_val'] \
            if 'trials_use_each_prf_val' in kwargs.keys() else None
        
        print('initializing for validation'); sys.stdout.flush()
        self.__init_for_val__(voxel_data_val, image_inds_val); 
        print('done initializing for validation'); sys.stdout.flush()
        
        with torch.no_grad(): # make sure local gradients are off to save memory
        
            # First looping over pRFs - there are fewer pRFs than voxels, so this will be faster 
            # than looping over voxels first would be.
            self.feature_loader.clear_big_features()
            
            for mm in range(self.n_prfs):
                
                if mm>1 and self.debug:
                    break
                if not np.any(self.best_prf_models==mm):
                    print('No voxels have this pRF as their best model, skipping it.')
                    continue
                    
                print('Getting features for prf %d of %d'%(mm, self.n_prfs)); 
                sys.stdout.flush()

                self.__val_one_prf__(mm)   
          
            self.feature_loader.clear_big_features()
            
        self.voxel_data_val = None
        self.image_inds_val = None
        self.shuff_inds_val = None
        gc.collect()
        
    def __init_for_val__(self, voxel_data_val, \
                             image_inds_val, \
                             **kwargs):
        
        self.voxel_data_val = voxel_data_val;
        self.image_inds_val = image_inds_val;
        self.n_voxels = self.voxel_data_val.shape[1]
        
        self.n_val_trials = len(self.image_inds_val)
        assert(self.voxel_data_val.shape[1]==self.n_voxels)

        print('about to preallocate arrays'); sys.stdout.flush()
        # val_cc is the correlation coefficient bw real and predicted responses across trials, for each voxel.
        if self.shuffle_data:
            self.val_cc  = np.zeros(shape=(self.n_voxels, self.n_partial_versions, self.n_shuff_iters), \
                                    dtype=self.dtype)
            self.val_r2 = np.zeros(shape=(self.n_voxels, self.n_partial_versions, self.n_shuff_iters), \
                                   dtype=self.dtype)
        else:
            self.val_cc  = np.zeros(shape=(self.n_voxels, self.n_partial_versions), dtype=self.dtype)
            self.val_r2 = np.zeros(shape=(self.n_voxels, self.n_partial_versions), dtype=self.dtype)

        # Saving full trial-by-trial feature activations, for each pRF.
        # Need these for later analyses.
        self.features_each_prf = np.full(fill_value=0, shape=(self.n_val_trials, self.max_features, self.n_prfs), \
                                    dtype=self.dtype)
        
        # Saving full trial-by-trial predictions for each voxel, each partial model.
        # Need these for later analyses.
        self.pred_voxel_data = np.full(fill_value=0, shape=(self.n_val_trials, self.n_voxels, self.n_partial_versions), \
                                  dtype=self.dtype)
        print('made arrays'); sys.stdout.flush()
        if self.shuffle_data:
            # prep for permutation test evaluation, if doing 
            if self.trials_use_each_prf_val is not None:
                raise ValueError('cannot specify a trial subset when also doing permutation test')
            if self.shuff_rnd_seed is not None:
                np.random.seed(self.shuff_rnd_seed)    
            print('making shuffle inds')
            self.shuff_inds_val = np.zeros((self.n_val_trials,self.n_shuff_iters),dtype=int)
            for xx in range(self.n_shuff_iters):
                self.shuff_inds_val[:,xx] = np.random.permutation(self.n_val_trials)
                
    def __val_one_prf__(self, mm):

        
        voxels_to_do = np.where(self.best_prf_models==mm)[0]  
        n_voxel_batches = int(np.ceil(len(voxels_to_do)/self.voxel_batch_size))

        print('about to load features') sys.stdout.flush()
        # all_feat_concat is size [ntrials x nfeatures] (where nfeatures can be <max_features)
        # feature_inds_defined is [max_features]
        all_feat_concat, feature_inds_defined = self.feature_loader.load(self.image_inds_val, mm, fitting_mode=False)
        print('loaded features'); sys.stdout.flush()
              
        if self.zscore:
            # using mean and std that were computed on training set during fitting - keeping 
            # these pars constant here seems to improve fits. 
            tiled_mean = np.tile(self.features_mean[mm,feature_inds_defined], [self.n_val_trials, 1])
            tiled_std = np.tile(self.features_std[mm,feature_inds_defined], [self.n_val_trials, 1])
            all_feat_concat = (all_feat_concat - tiled_mean)/tiled_std
            assert(not np.any(np.isnan(all_feat_concat)) and not np.any(np.isinf(all_feat_concat)))

        # saving all these features for use later on
        self.features_each_prf[:,feature_inds_defined,mm] = all_feat_concat
        all_feat_concat = None;
        gc.collect()
        
              
        # get data ready to do validation
        features_full = self.features_each_prf[:,:,mm:mm+1]

        if self.trials_use_each_prf_val is not None:
            # select subset of trials to work with
            trials_use = self.trials_use_each_prf_val[:,mm]
            features_full = features_full[trials_use,:,:]
            voxel_data_use = self.voxel_data_val[trials_use,:]
            if np.sum(trials_use)==0:
                print('prf %d: no trials are included here, skipping validation for all voxels with this pRF!'%mm)
                self.val_cc[voxels_to_do,:] = np.nan
                self.val_r2[voxels_to_do,:] = np.nan
                return
        else:
            trials_use = np.ones((self.n_val_trials,),dtype=bool)
            voxel_data_use = self.voxel_data_val

        print('prf %d: using %d validation set trials'%(mm, voxel_data_use.shape[0]))
        sys.stdout.flush()
        
        # Next looping over all voxels with this same pRF, in batches        
        for vv in range(n_voxel_batches):

            print('Getting predictions for voxel batch %d of %d'%(vv, n_voxel_batches))
            sys.stdout.flush()
        
            vinds = np.arange(self.voxel_batch_size*vv, np.min([self.voxel_batch_size*(vv+1), len(voxels_to_do)]))
            voxel_batch_inds = voxels_to_do[vinds]
           
            # double check the pRF estimates are same
            assert(np.all(self.best_prf_models[voxel_batch_inds,:]==mm))

            if vv>1 and self.debug:
                continue

            # Looping over versions of model w different features set to zero (variance partition)
            for pp in range(self.n_partial_versions):

                print('\nEvaluating version %d of %d: %s'%(pp, self.n_partial_versions, self.partial_version_names[pp]))
                sys.stdout.flush()
        
                # masks describes the indices of the features that are included in this partial model
                # n_features_max in length
                features_to_use = self.masks[0:self.max_features,pp]==1
                print('Includes %d features'%np.sum(features_to_use))
                sys.stdout.flush()
        
                # Take out the relevant features now
                features = np.tile(features_full[:,features_to_use,:], [1,1,len(voxel_batch_inds)])
                # Note there may be some zeros in this matrix, if we used fewer than the 
                # max number of features.
                # But they are zero in weight matrix too, so turns out ok.

                if not self.shuffle_data:
                    self.__get_preds_one_batch__(features, \
                                                 features_to_use, \
                                                 trials_use, \
                                                 voxel_data_use, 
                                                 voxel_batch_inds, pp)
                else:
                    self.__get_preds_one_batch_shuffle__(features, \
                                                 features_to_use, \
                                                 trials_use, \
                                                 voxel_data_use, 
                                                 voxel_batch_inds, pp)
                          
                        
    def __get_preds_one_batch__(self, features, \
                                         features_to_use, \
                                         trials_use, \
                                         voxel_data_use, 
                                         voxel_batch_inds, pp):

        _weights = torch_utils._to_torch(self.best_weights[voxel_batch_inds,:,pp], device=self.device)   
        _weights = _weights[:, features_to_use]
        _bias = torch_utils._to_torch(self.best_biases[voxel_batch_inds,pp], device=self.device)

        n_trials_use = np.sum(trials_use)
        pred_block = np.full(fill_value=0, shape=(n_trials_use, len(voxel_batch_inds)), dtype=self.dtype)

        # Now looping over validation set trials in batches
        n_trial_batches = int(np.ceil(n_trials_use/self.sample_batch_size))
        for ti in range(n_trial_batches):

            trial_batch_inds = np.arange(self.sample_batch_size*ti, np.min([self.sample_batch_size*(ti+1), n_trials_use]))

            # features is [trials x features x voxels]
            _features = torch_utils._to_torch(features[trial_batch_inds,:,:], device=self.device)
            # swap dims to [voxels x trials x features]
            _features = torch.transpose(torch.transpose(_features, 0, 2), 1, 2)
            # weights is [voxels x features]
            # _r will be [voxels x trials x 1] - then [trials x voxels]
            _r = torch.squeeze(torch.bmm(_features, torch.unsqueeze(_weights, 2)), dim=2).t() 

            if _bias is not None:
                _r = _r + torch.tile(torch.unsqueeze(_bias, 0), [_r.shape[0],1])

            pred_block[trial_batch_inds] = torch_utils.get_value(_r) 

        # Now for this batch of voxels and this partial version of the model, measure performance.
        self.val_cc[voxel_batch_inds,pp] = stats_utils.get_corrcoef(voxel_data_use[:,voxel_batch_inds], pred_block)
        self.val_r2[voxel_batch_inds,pp] = stats_utils.get_r2(voxel_data_use[:,voxel_batch_inds], pred_block)

        # Make sure to save the trial-wise predictions, for use in analyses later on 
        pred_these_trials = self.pred_voxel_data[trials_use,:,pp]
        pred_these_trials[:,voxel_batch_inds] = pred_block
        self.pred_voxel_data[trials_use,:,pp] = pred_these_trials

        sys.stdout.flush()

            
        
    def __get_preds_one_batch_shuffle__(self, features, \
                                         features_to_use, \
                                         trials_use, \
                                         voxel_data_use, 
                                         voxel_batch_inds, pp):

        # weights is [voxels x features x n_shuff_iters]
        weights = self.best_weights[voxel_batch_inds,:,pp,:][:,features_to_use, :]
        # biases is [voxels x n_shuff_iters]
        bias = self.best_biases[voxel_batch_inds,pp,:]
        
        n_trials_use = np.sum(trials_use)
        assert(n_trials_use==features.shape[0]) 
        pred_block = np.full(fill_value=0, shape=(n_trials_use, len(voxel_batch_inds), self.n_shuff_iters), dtype=self.dtype)

        # Now looping over validation set trials in batches
        n_trial_batches = int(np.ceil(n_trials_use/self.sample_batch_size))
        for ti in range(n_trial_batches):
            
            trial_batch_inds = np.arange(self.sample_batch_size*ti, np.min([self.sample_batch_size*(ti+1), n_trials_use]))

            _features = torch_utils._to_torch(features[trial_batch_inds,:,:], device=self.device)
            # features is [#samples, #features, #voxels]
            # swap dims to [#voxels, #samples, features]
            _features = torch.transpose(torch.transpose(_features, 0, 2), 1, 2)

            r_all = np.zeros((len(voxel_batch_inds), len(trial_batch_inds), self.n_shuff_iters), dtype=self.dtype)
           
            # also batching over shuffle iterations
            for bb, batch_inds in enumerate(self.shuff_batch_inds):

                print('permutation test, batch %d of %d'%(bb, len(self.shuff_batch_inds)))

                _weights = torch_utils._to_torch(weights[:,:,batch_inds], device=self.device)
                _bias = torch_utils._to_torch(bias[:,batch_inds], device=self.device)
                
                # _r will be [voxels x samples x n_shuff_iters]
                _r = torch.bmm(_features, _weights)

                if _bias is not None:
                    _r = _r + torch.tile(torch.unsqueeze(_bias, 1), [1,_r.shape[1],1])

                r_all[:,:,batch_inds] = torch_utils.get_value(_r)
            
            pred_block[trial_batch_inds,:,:] = np.moveaxis(r_all, [0], [1])
            # pred_block[trial_batch_inds,:,:] = torch_utils.get_value(torch.transpose(_r, 0,1 ))

        # Now for this batch of voxels and this partial version of the model, measure performance.
        for xx in range(self.n_shuff_iters):
            # use the randomized validation set order here
            shuff_order = self.shuff_inds_val[:,xx]
            shuff_dat = voxel_data_use[:,voxel_batch_inds][shuff_order,:]
            self.val_cc[voxel_batch_inds,pp,xx] = stats_utils.get_corrcoef(shuff_dat, pred_block[:,:,xx])
            self.val_r2[voxel_batch_inds,pp,xx] = stats_utils.get_r2(shuff_dat, pred_block[:,:,xx])

        # We don't need to save every trial-wise prediction here because they'll get very large.
        # just save the first one in case we want to check values later.
        pred_these_trials = self.pred_voxel_data[trials_use,:,pp]
        pred_these_trials[:,voxel_batch_inds] = pred_block[:,:,0]
        self.pred_voxel_data[trials_use,:,pp] = pred_these_trials

        sys.stdout.flush()
        
        r_all = None;
        pred_block = None;
        gc.collect()





