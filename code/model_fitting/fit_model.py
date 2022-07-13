"""
Run the model fitting for FWRF model. 
There are a few different versions of fitting in this script, the input arguments tell which kind of fitting to do.
"""

# import basic modules
import sys
import os
import time
import numpy as np
import argparse
import gc

# import custom modules
from utils import nsd_utils, roi_utils, default_paths

import initialize_fitting, arg_parser, fwrf_model
from analyze_fits import feature_selectivity, semantic_selectivity
# import fwrf_fit, fwrf_predict, 

try:
    device = initialize_fitting.init_cuda()
except:
    device = 'cpu:0'
    
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

print('numpy version: %s'%np.__version__)
#################################################################################################
        
    
def fit_fwrf(args):

    model_name, fitting_types = initialize_fitting.get_full_save_name(args)
    output_dir, fn2save = initialize_fitting.get_save_path(model_name, args)
    sys.stdout.flush()
    
    def save_all(fn2save):
    
        """
        Define all the important parameters that have to be saved
        """
        dict2save = {
        'subject': args.subject,
        'volume_space': args.volume_space,
        'fitting_types': fitting_types, 
        'voxel_mask': voxel_mask,
        'average_image_reps': args.average_image_reps, 
        'brain_nii_shape': brain_nii_shape,
        'image_order': image_order,
        'voxel_index': voxel_index,
        'voxel_roi': voxel_roi,
        'voxel_ncsnr': voxel_ncsnr, 
        'which_prf_grid': args.which_prf_grid,
        'prfs_fixed_sigma': args.prf_fixed_sigma, 
        'models': prf_models,        
        'best_losses': best_losses,           
        'best_lambdas': best_lambdas,
        'best_params': best_params,       
        'lambdas': lambdas, 
        'val_cc': val_cc,
        'val_r2': val_r2,    
        'partial_masks': partial_masks, 
        'partial_version_names': partial_version_names,        
        'zscore_features': args.zscore_features, 
        'ridge': args.ridge,
        'set_lambda_per_group': args.set_lambda_per_group, 
        'debug': args.debug,
        'up_to_sess': args.up_to_sess,
        'single_sess': args.single_sess,
        'use_precomputed_prfs': args.use_precomputed_prfs,
        'saved_prfs_fn': saved_prfs_fn,
        'best_layer_each_voxel': best_layer_each_voxel,
        'saved_best_layer_fn': saved_best_layer_fn,
        'voxel_subset_is_done_trn': voxel_subset_is_done_trn,
        'voxel_subset_is_done_val': voxel_subset_is_done_val,
        'trial_subset': args.trial_subset, 
        'do_corrcoef': args.do_corrcoef,
        }
        # Might be some more things to save, depending what kind of fitting this is
        if args.use_model_residuals:
            dict2save.update({
            'residuals_model_name': args.residuals_model_name, 
            'residuals_model_filename': residuals_model_filename,
            })
        if args.do_tuning:
            dict2save.update({
            'corr_each_feature': corr_each_feature
            })
        if args.do_sem_disc:
            dict2save.update({
            'sem_discrim_each_axis': sem_discrim_each_axis,
            'sem_corr_each_axis': sem_corr_each_axis,
            'discrim_type_list': discrim_type_list,
            'n_sem_samp_each_axis': n_sem_samp_each_axis,
            'mean_each_sem_level': mean_each_sem_level,
            'axes_to_do': axes_to_do, 
            'sem_partial_corrs': sem_partial_corrs, 
            'sem_partial_n_samp': sem_partial_n_samp, 
            'axes_to_balance': axes_to_balance,
            'sem_discrim_each_axis_balanced': sem_discrim_each_axis_balanced,
            'sem_corr_each_axis_balanced': sem_corr_each_axis_balanced,           
            'n_sem_samp_each_axis_balanced': n_sem_samp_each_axis_balanced,
            'mean_each_sem_level_balanced': mean_each_sem_level_balanced,
            })
        if args.shuffle_data:
            dict2save.update({
            'n_shuff_iters': args.n_shuff_iters, 
            'shuff_rnd_seed': args.shuff_rnd_seed,
            'shuff_batch_size': args.shuff_batch_size,
            'voxel_batch_size_outer': args.voxel_batch_size_outer,
            })
        if args.bootstrap_data:
            dict2save.update({
            'n_boot_iters': args.n_boot_iters, 
            'boot_rnd_seed': args.boot_rnd_seed,
            'boot_val_only': args.boot_val_only,
            'voxel_batch_size_outer': args.voxel_batch_size_outer,
            })
        if np.any(['semantic' in ft for ft in fitting_types]):
            dict2save.update({
            'semantic_feature_set': args.semantic_feature_set,
            })
        if np.any(['sketch_tokens' in ft for ft in fitting_types]):
            dict2save.update({         
            'use_pca_st_feats': args.use_pca_st_feats,
            'use_residual_st_feats': args.use_residual_st_feats,
            })          
        if np.any(['pyramid' in ft for ft in fitting_types]):
            dict2save.update({
            'pyr_pca_type': args.pyr_pca_type,
            'group_all_hl_feats': args.group_all_hl_feats,
            'do_pyr_varpart': args.do_pyr_varpart,
            })            
        if np.any(['gabor' in ft for ft in fitting_types]):
            dict2save.update({
            'n_ori_gabor': args.n_ori_gabor,
            'n_sf_gabor': args.n_sf_gabor,
            'gabor_nonlin_fn': args.gabor_nonlin_fn,
            'use_pca_gabor_feats': args.use_pca_gabor_feats,
            })
        if np.any(['alexnet' in ft for ft in fitting_types]):
            dict2save.update({
            'alexnet_layer_name': args.alexnet_layer_name,
            'alexnet_padding_mode': args.alexnet_padding_mode,
            'use_pca_alexnet_feats': args.use_pca_alexnet_feats, 
            })
        if np.any(['clip' in ft for ft in fitting_types]):
            dict2save.update({
            'clip_layer_name': args.clip_layer_name,
            'clip_model_architecture': args.clip_model_architecture,
            'use_pca_clip_feats': args.use_pca_clip_feats,   
            })

        print('\nSaving to %s\n'%fn2save)
        print(dict2save.keys())
        np.save(fn2save, dict2save, allow_pickle=True)

    if (args.from_scratch) and not (args.date_str==0 or args.date_str=='0' or args.date_str==''):
        raise ValueError('if --from_scratch=True, should specify --date_str=0 (rather than entering a date)')    
    if (args.do_sem_disc or args.do_tuning) and not args.do_val:
        raise ValueError('to do tuning analysis or semantic discriminability, need to run validation (--do_val=True)')       
    if args.use_model_residuals and len(args.residuals_model_name)==0:
        raise ValueError('must specify the name of model the residuals are from')
        
    if args.shuffle_data or args.bootstrap_data:
        # for permutation test to work, need these conditions met (haven't tested otherwise)
        assert args.from_scratch
        assert not (args.do_sem_disc  or args.do_tuning)
        assert args.use_precomputed_prfs
        assert not (args.shuffle_data and args.bootstrap_data)
        
    val_r2 = None; 
    val_cc = None;
    
    corr_each_feature = None
    
    sem_discrim_each_axis = None
    sem_corr_each_axis = None
    discrim_type_list = None
    n_sem_samp_each_axis = None
    mean_each_sem_level = None
    
    axes_to_do = None
    sem_partial_corrs = None
    sem_partial_n_samp = None
    
    axes_to_balance = None
    sem_discrim_each_axis_balanced = None
    sem_corr_each_axis_balanced = None
    n_sem_samp_each_axis_balanced = None
    mean_each_sem_level_balanced = None
        
    if np.any(['alexnet' in ft for ft in fitting_types]):
        dnn_model='alexnet'
        n_dnn_layers = 5;
        assert(not np.any(['clip' in ft for ft in fitting_types]))
    elif np.any(['clip' in ft for ft in fitting_types]):
        dnn_model='clip'
        n_dnn_layers = 16;
        assert(not np.any(['alexnet' in ft for ft in fitting_types]))
    else:
        dnn_model = None
          
    
    ########## LOADING THE DATA #############################################################################
    # decide what voxels to use  
    voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = \
                                roi_utils.get_voxel_roi_info(args.subject, \
                                args.volume_space)

    if (args.single_sess is not None) and (args.single_sess!=0):
        sessions = np.array([args.single_sess])
    else:
        sessions = np.arange(0,args.up_to_sess)
    # Get all data and corresponding images, in two splits. Always a fixed set that gets left out
    if not args.use_model_residuals:
        # normal case, using voxel data to fit model
        voxel_data, image_order, val_inds, holdout_inds, session_inds = \
                                    nsd_utils.get_data_splits(args.subject, \
                                    sessions=sessions, \
                                    voxel_mask=voxel_mask, volume_space=args.volume_space, \
                                    zscore_betas_within_sess=True, \
                                    shuffle_images=args.shuffle_images_once,\
                                    average_image_reps=args.average_image_reps, \
                                    random_voxel_data=args.random_voxel_data)
        
    else:
        # special case, using the residuals of another encoding model as input voxel data.
        voxel_data, image_order, val_inds, holdout_inds, session_inds, residuals_model_filename = \
                                    initialize_fitting.load_model_residuals(args, sessions)
    voxel_data_val = voxel_data[val_inds,:]
    voxel_data_trn = voxel_data[~val_inds & ~holdout_inds,:]
    voxel_data_holdout = voxel_data[holdout_inds,:]
    image_inds_val = image_order[val_inds]
    image_inds_trn = image_order[~val_inds & ~holdout_inds]
    image_inds_holdout = image_order[holdout_inds]
    
    if session_inds is not None:
        session_inds_val = session_inds[val_inds]
        session_inds_trn = session_inds[~val_inds & ~holdout_inds]
        session_inds_holdout = session_inds[holdout_inds]
    n_voxels = voxel_data_trn.shape[1]   

    ########## DEFINE PARAMETERS #############################################################################
    
    lambdas = initialize_fitting.get_lambdas(fitting_types=fitting_types, \
                                             zscore_features=args.zscore_features, ridge=args.ridge)
    prf_models = initialize_fitting.get_prf_models(which_grid=args.which_prf_grid, verbose=True) 
    n_prfs = prf_models.shape[0]
    
    if args.prf_fixed_sigma is not None:
        assert(args.use_precomputed_prfs==False)
        # special case, we want to test all centers for only one size value.
        print('going to fix sigma at %.3f for all voxels'%args.prf_fixed_sigma)
        prfs_fit_mask = np.round(prf_models[:,2],3)==args.prf_fixed_sigma
        assert(np.sum(prfs_fit_mask)>=132)
        print('there are %d pRFs with this sigma value'%np.sum(prfs_fit_mask))
    else:
        prfs_fit_mask=None
    
    sys.stdout.flush()
    
    if (args.trial_subset!='all'):
        
        if not args.debug:
            assert(args.up_to_sess==40)
        assert(args.average_image_reps==True)
        
        trn_trials_use, holdout_trials_use, val_trials_use = \
                initialize_fitting.get_subsampled_trial_order(image_inds_trn, \
                                                              image_inds_holdout, \
                                                              image_inds_val, \
                                                              args=args, \
                                                              index=0)
        
        print('min trn trials: %d'%np.min(np.sum(trn_trials_use, axis=0)))
        print('min holdout trials: %d'%np.min(np.sum(holdout_trials_use, axis=0)))
        print('min val trials: %d'%np.min(np.sum(val_trials_use, axis=0)))
        
    else:
        trn_trials_use = None
        holdout_trials_use = None
        val_trials_use = None
   
        
    if args.use_precomputed_prfs:
        # If we already computed pRFs for this subject on some model, can load those now and use them during 
        # fitting. Faster than fitting pRFs each time.
        best_model_each_voxel, saved_prfs_fn = initialize_fitting.load_precomputed_prfs(args.subject, args)
        assert(len(best_model_each_voxel)==n_voxels)
    else:
        # otherwise fitting all params from scratch.
        best_model_each_voxel = None
        saved_prfs_fn = None

    
    ####### DEFINE VOXEL SUBSETS TO LOOP OVER ###############################################################
    # only used for clip/alexnet when layer_name is "best_layer", since diff voxels get fit w different features
    # otherwise this loop only goes once and voxel_subset_mask is all ones.
    
    if dnn_model is not None and (args.alexnet_layer_name=='best_layer' or args.clip_layer_name=='best_layer'):
        # special case, going to fit groups of voxels separately according to which dnn layer was best
        # creating a list of voxel masks here that will define the subsets to loop over.
        best_layer_each_voxel, saved_best_layer_fn = \
                  initialize_fitting.load_best_model_layers(args.subject, dnn_model)
        voxel_subset_masks = [best_layer_each_voxel==ll for ll in range(n_dnn_layers)]
        assert(len(best_layer_each_voxel)==n_voxels)
        assert(not args.save_model_residuals)
        assert(not args.shuffle_data and not args.bootstrap_data) 
        
        # Create feature loaders here
        feat_loader_full_list = [initialize_fitting.make_feature_loaders(args, fitting_types, vi=ll) \
                            for ll in range(n_dnn_layers)]
    elif args.shuffle_data or args.bootstrap_data:
        best_layer_each_voxel = None;
        saved_best_layer_fn = None;
        # to prevent running out of memory, i'm batching the voxels at this stage 
        # (doing all the shuffle iterations at once, for each batch)       
        bs = args.voxel_batch_size_outer
        n_batches = int(np.ceil(n_voxels/bs))
        voxel_subset_masks = [(np.arange(n_voxels)>=(nn*bs)) & (np.arange(n_voxels)<((nn+1)*bs)) \
                        for nn in range(n_batches)]
        feat_loader_full_list = [initialize_fitting.make_feature_loaders(args, fitting_types, vi=0) \
                        for nn in range(n_batches)]
        
    else:
        # going to fit all voxels w same model
        voxel_subset_masks = [np.ones((n_voxels,), dtype=bool)]
        best_layer_each_voxel = None;
        saved_best_layer_fn = None;
        
        # Create feature loaders here
        feat_loader_full_list = [initialize_fitting.make_feature_loaders(args, fitting_types, vi=0)]
        
    max_features_overall = np.max([fl.max_features for fl in feat_loader_full_list])      
    
    # getting info about how variance partition will be set up
    partial_masks_tmp, partial_version_names = feat_loader_full_list[0].get_partial_versions()      
    print(partial_version_names)
    print(np.sum(partial_masks_tmp, axis=1))
    n_partial_versions = len(partial_version_names)
    partial_masks = [[] for ii in range(len(voxel_subset_masks))] 
    for vi, fl in enumerate(feat_loader_full_list):
        partial_masks_tmp, partial_version_names = fl.get_partial_versions()         
        partial_masks[vi] = partial_masks_tmp
        assert(len(partial_version_names)==n_partial_versions)
        
                                 
    ###### LOAD LAST SAVED MODEL ############################################################################
    if not args.from_scratch:
        
        # stuff that needs to happen if we are resuming from some intermediate point
        print('\nLoading the results of training from %s\n'%fn2save)
        last_saved = np.load(fn2save, allow_pickle=True).item()
        # make sure that training was actually done, otherwise should start over 
        assert(np.any(last_saved['voxel_subset_is_done_trn']))
        assert(last_saved['up_to_sess']==args.up_to_sess)
        assert(last_saved['debug']==args.debug)
        assert(last_saved['which_prf_grid']==args.which_prf_grid)
        assert(np.all(last_saved['lambdas']==lambdas))
        print(last_saved['saved_best_layer_fn'])
        print(saved_best_layer_fn)
        print(last_saved['saved_best_layer_fn'].split('/')[7])
        print(saved_best_layer_fn.split('/')[7])
        assert(last_saved['saved_prfs_fn'].split('/')[7]==saved_prfs_fn.split('/')[7])
        assert(last_saved['saved_best_layer_fn'].split('/')[7]==saved_best_layer_fn.split('/')[7])
        assert('shuffle_data' not in last_saved.keys() or last_saved['shuffle_data']==False)
        
        voxel_subset_is_done_trn = last_saved['voxel_subset_is_done_trn']
        voxel_subset_is_done_val = last_saved['voxel_subset_is_done_val']
        
        if args.do_tuning:
            if 'corr_each_feature' in last_saved.keys() and last_saved['corr_each_feature'] is not None:
                corr_each_feature = last_saved['corr_each_feature']
            else:
                voxel_subset_is_done_val = np.zeros(np.shape(voxel_subset_is_done_val), dtype=bool)
        else:
            assert('corr_each_feature' not in last_saved.keys() or last_saved['corr_each_feature'] is None)
            
        if args.do_sem_disc:
            if (not args.overwrite_sem_disc) \
               and 'sem_discrim_each_axis' in last_saved.keys() \
               and last_saved['sem_discrim_each_axis'] is not None:
                sem_discrim_each_axis = last_saved['sem_discrim_each_axis']
                sem_corr_each_axis = last_saved['sem_corr_each_axis']
                discrim_type_list = last_saved['discrim_type_list']
                n_sem_samp_each_axis = last_saved['n_sem_samp_each_axis']
                mean_each_sem_level = last_saved['mean_each_sem_level']
                
                axes_to_do = last_saved['axes_to_do']
                sem_partial_corrs = last_saved['sem_partial_corrs']
                sem_partial_n_samp = last_saved['sem_partial_n_samp']
                
                axes_to_balance = last_saved['axes_to_balance']
                sem_discrim_each_axis_balanced = last_saved['sem_discrim_each_axis_balanced']
                sem_corr_each_axis_balanced = last_saved['sem_corr_each_axis_balanced']               
                n_sem_samp_each_axis_balanced = last_saved['n_sem_samp_each_axis_balanced']
                mean_each_sem_level_balanced = last_saved['mean_each_sem_level_balanced']
            else:
                voxel_subset_is_done_val = np.zeros(np.shape(voxel_subset_is_done_val), dtype=bool)
        else:
            assert('sem_discrim_each_axis' not in last_saved.keys() or last_saved['sem_discrim_each_axis'] is None)
                

        print('training done for subsets:')
        print(voxel_subset_is_done_trn)
        print('validation done for subsets:')
        print(voxel_subset_is_done_val)
        best_losses = last_saved['best_losses']
        best_lambdas = last_saved['best_lambdas']
        best_params = last_saved['best_params'] 
        best_prf_model_pars, best_weights, best_biases, \
                           features_mean, features_std, best_prf_models = best_params
        val_cc = last_saved['val_cc']
        val_r2 = last_saved['val_r2']
      
    else:
        voxel_subset_is_done_trn = np.zeros((len(voxel_subset_masks),),dtype=bool)
        voxel_subset_is_done_val = np.zeros((len(voxel_subset_masks),),dtype=bool)
        
        # preallocate some arrays for params over all voxels
        best_losses = np.zeros((n_voxels, n_partial_versions), dtype=np.float32)
        best_lambdas = np.zeros((n_voxels, n_partial_versions), dtype=int)
        
        best_prf_models = np.zeros((n_voxels, n_partial_versions), dtype=int)
        features_mean = np.zeros((n_prfs, max_features_overall,len(voxel_subset_masks)), dtype=np.float32)
        features_std = np.zeros((n_prfs, max_features_overall,len(voxel_subset_masks)), dtype=np.float32)
        if args.shuffle_data or args.bootstrap_data:
            if args.shuffle_data:
                it = args.n_shuff_iters
            else:
                it = args.n_boot_iters
            val_cc = np.zeros((n_voxels, n_partial_versions, it), dtype=np.float32)
            val_r2 = np.zeros((n_voxels, n_partial_versions, it), dtype=np.float32) 
            best_weights = None # save memory by not permanently saving the weights...
            best_biases = None
        else:            
            val_cc = np.zeros((n_voxels, n_partial_versions), dtype=np.float32)
            val_r2 = np.zeros((n_voxels, n_partial_versions), dtype=np.float32) 
            best_weights = np.zeros((n_voxels, max_features_overall, n_partial_versions), dtype=np.float32)
            best_biases = np.zeros((n_voxels, n_partial_versions), dtype=np.float32)
        
    ########### LOOPING OVER VOXEL SUBSETS ######################################################
    for vi, voxel_subset_mask in enumerate(voxel_subset_masks):
        
        voxel_data_trn_use = voxel_data_trn[:,voxel_subset_mask]
        voxel_data_holdout_use = voxel_data_holdout[:,voxel_subset_mask]
        voxel_data_val_use = voxel_data_val[:,voxel_subset_mask]
        if best_model_each_voxel is not None:
            best_model_each_voxel_use = best_model_each_voxel[voxel_subset_mask]
        else:
            best_model_each_voxel_use = None
        print('\nStarting fitting for voxel mask %d of %d, number of voxels this loop=%d'%(vi, \
                                           len(voxel_subset_masks), voxel_data_trn_use.shape[1]))
        if voxel_data_trn_use.shape[1]==0:
            print('no voxels, continuing loop')
            voxel_subset_is_done_trn = True
            continue
        
        # pull out my current feature loader
        feat_loader_full = feat_loader_full_list[vi]
        max_features = feat_loader_full.max_features 
        
        
        ########## INITIALIZE ENCODING MODEL ##################################################
        
        model = fwrf_model.encoding_model(feat_loader_full, lambdas=lambdas, \
                                            best_model_each_voxel = best_model_each_voxel_use, \
                                            zscore=args.zscore_features, \
                                            add_bias=True, \
                                            set_lambda_per_group = args.set_lambda_per_group, \
                                            voxel_batch_size=args.voxel_batch_size,\
                                            sample_batch_size=args.sample_batch_size,\
                                            device=device,\
                                            prfs_fit_mask = prfs_fit_mask, \
                                            shuffle_data = args.shuffle_data, \
                                            shuff_rnd_seed = args.shuff_rnd_seed, \
                                            n_shuff_iters = args.n_shuff_iters, \
                                            shuff_batch_size = args.shuff_batch_size, \
                                            bootstrap_data = args.bootstrap_data, \
                                            boot_rnd_seed = args.boot_rnd_seed, \
                                            n_boot_iters = args.n_boot_iters, \
                                            boot_val_only = args.boot_val_only, \
                                            do_corrcoef = args.do_corrcoef, \
                                            dtype=np.float32, debug=args.debug)
                  
          
        ########### FIT ENCODING MODEL ###################################################################
        
        if not voxel_subset_is_done_trn[vi]:
   
            print('\nStarting training (voxel subset %d of %d)...\n'%(vi, len(voxel_subset_masks)))
            print(len(image_inds_trn))

            
            model.fit(image_inds_trn, voxel_data_trn_use, \
                        image_inds_holdout, voxel_data_holdout_use,\
                        trials_use_each_prf_trn = trn_trials_use,\
                        trials_use_each_prf_holdout = holdout_trials_use)
            print('done with fitting')
          
            # taking the fit params for this set of voxels and putting them into the full array over all voxels
            best_losses[voxel_subset_mask,:] = model.best_losses
            best_lambdas[voxel_subset_mask,:] = model.best_lambdas 
            features_mean[:,0:max_features,vi] = model.features_mean
            features_std[:,0:max_features,vi] = model.features_std
            best_prf_models[voxel_subset_mask,:] = model.best_prf_models
            if not args.shuffle_data and not args.bootstrap_data:
                best_weights[voxel_subset_mask,0:max_features,:] = model.best_weights
                best_biases[voxel_subset_mask,:] = model.best_biases
            
            model.best_lambdas = None;
            model.best_losses = None;
            gc.collect()
            sys.stdout.flush()

            voxel_subset_is_done_trn[vi] = True
            print('done with params')
        else:
            best_losses_tmp = best_losses[voxel_subset_mask,:]
            best_lambdas_tmp = best_lambdas[voxel_subset_mask,:]
            best_weights_tmp = best_weights[voxel_subset_mask,0:max_features,:]
            best_biases_tmp = best_biases[voxel_subset_mask,:]
            best_prf_models_tmp = best_prf_models[voxel_subset_mask,:]
            features_mean_tmp = features_mean[:,0:max_features,vi]
            features_std_tmp = features_std[:,0:max_features,vi]
            
            # put these into the model so that we can evaluate it later
            model.best_weights, model.best_biases, \
            model.best_prf_models, \
            model.features_mean, model.features_std = \
                best_weights_tmp, best_biases_tmp, \
                best_prf_models_tmp, \
                features_mean_tmp, features_std_tmp

        # this is the list of params that gets saved, make sure to update it each time
        if args.shuffle_data or args.bootstrap_data:
            # to keep the saved file from getting really big, only saving the weights for first 
            # permutation iteration.
            # this is why we can't resume from the middle of fitting.
            best_params = [prf_models[best_prf_models,:], None, None, \
                           features_mean, features_std, best_prf_models]
        else:
            best_params = [prf_models[best_prf_models,:], best_weights, best_biases, \
                           features_mean, features_std, best_prf_models]
        
        
        
        print('about to save')
                  
        save_all(fn2save)   
            
        ############### VALIDATE MODEL ##################################################################
    
        if args.do_val and not voxel_subset_is_done_val[vi]: 
            
            print('Starting validation (voxel subset %d of %d)...\n'%(vi, len(voxel_subset_masks)))
            sys.stdout.flush()
    
            model.validate(voxel_data_val_use, image_inds_val, trials_use_each_prf_val = val_trials_use)
        
            val_cc_tmp, val_r2_tmp, voxel_data_val_pred, features_each_prf =\
                model.val_cc, model.val_r2, model.pred_voxel_data, model.features_each_prf
                     
            val_cc[voxel_subset_mask,:] = val_cc_tmp
            val_r2[voxel_subset_mask,:] = val_r2_tmp
                                 
            if (not args.do_tuning) and (not args.do_sem_disc):
                voxel_subset_is_done_val[vi] = True
            save_all(fn2save) 
            
            ############# ESTIMATE FEATURE SELECTIVITY #########################################################
            sys.stdout.flush()
            if args.do_tuning:

                print('\nStarting feature tuning analysis (voxel subset %d of %d)...\n'%(vi, len(voxel_subset_masks)))
                sys.stdout.flush()
                corr_each_feature_tmp = feature_selectivity.get_feature_tuning(model.best_prf_models, features_each_prf, \
                                                                        voxel_data_val_pred, \
                                                                        trials_use_each_prf = val_trials_use, \
                                                                        debug=args.debug)
                if vi==0:
                    corr_each_feature = np.zeros((n_voxels, corr_each_feature_tmp.shape[1]), dtype=corr_each_feature_tmp.dtype)  
                max_features = feat_loader_full.max_features
                if corr_each_feature.shape[1]<max_features:
                    n2pad = max_features - corr_each_feature.shape[1]
                    print('padding by %d elements'%n2pad)
                    print(np.shape(corr_each_feature))
                    corr_each_feature = np.pad(corr_each_feature, [[0,0], [0, n2pad]])
                    print(np.shape(corr_each_feature))
                corr_each_feature[voxel_subset_mask,0:max_features] = corr_each_feature_tmp                
                if not args.do_sem_disc:
                    voxel_subset_is_done_val[vi] = True
                save_all(fn2save)

            ########### ESTIMATE SEMANTIC DISCRIMINABILITY #######################################################
            sys.stdout.flush()
            if args.do_sem_disc:
                
                print('\nStarting semantic discriminability analysis (voxel subset %d of %d)...\n'%(vi, len(voxel_subset_masks)))
                sys.stdout.flush()
                labels_all, discrim_type_list, unique_labs_each = \
                        initialize_fitting.load_labels_each_prf(args.subject, args.which_prf_grid,\
                                                        image_inds=image_inds_val, \
                                                        models=prf_models,verbose=False, \
                                                        debug=args.debug)
                discrim_tmp, corr_tmp, n_samp_tmp, mean_tmp = \
                        semantic_selectivity.get_semantic_discrim(model.best_prf_models, \
                                                          labels_all, unique_labs_each, \
                                                          voxel_data_val_pred,\
                                                          trials_use_each_prf = val_trials_use, \
                                                          debug=args.debug)
                if vi==0:
                    sem_discrim_each_axis = np.zeros((n_voxels, discrim_tmp.shape[1]), \
                                                     dtype=discrim_tmp.dtype) 
                    sem_corr_each_axis = np.zeros((n_voxels, corr_tmp.shape[1]), \
                                                     dtype=corr_tmp.dtype)
                    n_sem_samp_each_axis = np.zeros((n_voxels, n_samp_tmp.shape[1], n_samp_tmp.shape[2]), \
                                                     dtype=n_samp_tmp.dtype)
                    mean_each_sem_level = np.zeros((n_voxels, mean_tmp.shape[1], mean_tmp.shape[2]), \
                                                     dtype=mean_tmp.dtype)
                sem_discrim_each_axis[voxel_subset_mask,:] = discrim_tmp
                sem_corr_each_axis[voxel_subset_mask,:] = corr_tmp
                n_sem_samp_each_axis[voxel_subset_mask,:,:] = n_samp_tmp
                mean_each_sem_level[voxel_subset_mask,:,:] = mean_tmp
#                 voxel_subset_is_done_val[vi] = True
                save_all(fn2save)
    
                # compute partial correlations for some axes 
                axes_to_do = [0,2,3]
                print('\nGoing to compute partial correlations, for these pairs of axes:')
                print([discrim_type_list[aa] for aa in axes_to_do])
                partial_corr_tmp, n_samp_tmp = \
                        semantic_selectivity.get_semantic_partial_corrs(model.best_prf_models, \
                                                          labels_all, axes_to_do=axes_to_do, \
                                                          unique_labels_each=unique_labs_each, \
                                                          val_voxel_data_pred=voxel_data_val_pred,\
                                                          trials_use_each_prf = val_trials_use, \
                                                          debug=args.debug)
                if vi==0:                 
                    sem_partial_corrs = np.zeros((n_voxels, partial_corr_tmp.shape[1]), \
                                                     dtype=partial_corr_tmp.dtype)
                    sem_partial_n_samp = np.zeros((n_voxels, n_samp_tmp.shape[1], n_samp_tmp.shape[2]), \
                                                     dtype=n_samp_tmp.dtype)

                sem_partial_corrs[voxel_subset_mask,:] = partial_corr_tmp
                sem_partial_n_samp[voxel_subset_mask,:,:] = n_samp_tmp
        
                voxel_subset_is_done_val[vi] = True
                save_all(fn2save)
                
                
                
        if args.save_model_residuals:
                                 
            # going to compute model predictions on entire data set, and save them as a separate
            # file for use later on.
                                 
            model.validate(voxel_data, image_order, trials_use_each_prf_val = None)
        
            all_dat_r2 = model.val_r2
            all_dat_pred = model.pred_voxel_data
                                 
            initialize_fitting.save_model_residuals(voxel_data, all_dat_pred[:,:,0], \
                                                    output_dir, model_name, \
                                                    image_order, val_inds, \
                                                    session_inds, all_dat_r2[:,0], \
                                                    args)
            
                 
        # Done!

if __name__ == '__main__':
    
    args = arg_parser.get_args()
    fit_fwrf(args)
