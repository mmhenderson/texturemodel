import argparse
import numpy as np
import distutils.util
import time

def nice_str2bool(x):
    return bool(distutils.util.strtobool(x))

def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--volume_space", type=nice_str2bool, default=True,
                    help="want to do fitting with volume space or surface space data? 1 for volume, 0 for surface.")
    parser.add_argument("--up_to_sess", type=int,default=1,
                    help="analyze sessions 1-#")
    parser.add_argument("--single_sess", type=int, default=0,
                    help="analyze just this one session (enter integer)")
    parser.add_argument("--average_image_reps", type=nice_str2bool, default=1,
                    help="average over trial repetitions of same image?")
    parser.add_argument("--save_model_residuals", type=nice_str2bool, default=0, 
                    help="save model residuals for each voxel?")
    parser.add_argument("--use_model_residuals", type=nice_str2bool, default=0, 
                    help="load/fit model residuals for each voxel?")
    parser.add_argument("--residuals_model_name", type=str, default='', 
                    help="model the residuals are from?")
    
    parser.add_argument("--trial_subset", type=str,default='all', 
                    help="fit for a subset of trials only? default all trials")
   
    parser.add_argument("--image_set", type=str,default='none', 
                    help="if evaluating on an independent image set, what is it called?")
   
    parser.add_argument("--which_prf_grid", type=int,default=5,
                    help="which grid of candidate prfs?")
    parser.add_argument("--prf_fixed_sigma", type=float, default=None, 
                    help="if sigma is fixed, what sigma value to use?")
    
    parser.add_argument("--fitting_type", type=str,default='texture_pyramid',
                    help="what kind of fitting are we doing? opts are 'texture_pyramid', 'texture_gabor', 'gabor_solo'")
    parser.add_argument("--fitting_type2", type=str,default='',
                    help="additional fitting type, for variance partition?")
    parser.add_argument("--fitting_type3", type=str,default='',
                    help="additional fitting type, for variance partition?")
    
    parser.add_argument("--ridge", type=nice_str2bool, default=True,
                    help="want to do ridge regression (lambda>0)? 1 for yes, 0 for no")
    parser.add_argument("--set_lambda_per_group", type=nice_str2bool, default=False,
                    help="want to allow lambda to differ between diff feature groups?? 1 for yes, 0 for no")
    parser.add_argument("--zscore_features", type=nice_str2bool, default=True,
                    help="want to z-score each feature right before fitting encoding model? 1 for yes, 0 for no")
    parser.add_argument("--do_corrcoef", type=nice_str2bool, default=True,
                    help="want to compute validation set correlation coefficient, in addition to R2? 1 for yes, 0 for no")
    
    # these are ways of doing shuffling just once, as a quick test
    parser.add_argument("--shuffle_images_once", type=nice_str2bool,default=False,
                    help="want to shuffle the images randomly (control analysis)? 1 for yes, 0 for no")
    parser.add_argument("--random_images", type=nice_str2bool,default=False,
                    help="want to use random gaussian values for images (control analysis)? 1 for yes, 0 for no")
    parser.add_argument("--random_voxel_data", type=nice_str2bool,default=False,
                    help="want to use random gaussian values for voxel data (control analysis)? 1 for yes, 0 for no")
    
    # shuffle_data will actually compute multiple shuffling iterations, for permutation test.
    parser.add_argument("--shuffle_data", type=nice_str2bool,default=False,
                    help="want to run permutation test? 1 for yes, 0 for no")
    parser.add_argument("--n_shuff_iters", type=int,default=1000,
                    help="how many shuffle iters?")
    parser.add_argument("--shuff_batch_size", type=int,default=100,
                    help="batch size over permutation iterations")
    parser.add_argument("--shuff_rnd_seed", type=int,default=0,
                    help="random seed to use for shuffling in permutation test.")
   
   
    parser.add_argument("--bootstrap_data", type=nice_str2bool,default=False,
                    help="want to run bootstrap test? 1 for yes, 0 for no")
    parser.add_argument("--boot_val_only", type=nice_str2bool,default=False,
                    help="want to run bootstrapping just during validation (faster)? 1 for yes, 0 for no")
    parser.add_argument("--n_boot_iters", type=int,default=1000,
                    help="how many shuffle iters?")
    parser.add_argument("--boot_rnd_seed", type=int,default=0,
                    help="random seed to use for shuffling in bootstrap test.")
   
    
    parser.add_argument("--debug",type=nice_str2bool,default=False,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    parser.add_argument("--from_scratch", type=nice_str2bool,default=True,
                    help="want to do model training from the start? 1 for yes, 0 for no")
    parser.add_argument("--use_precomputed_prfs", type=nice_str2bool,default=False,
                    help="want to use prf estimates that were already computed? 1 for yes, 0 for no")
    parser.add_argument("--prfs_model_name", type=str, default='', 
                    help="model the prfs are from? by default, uses alexnet pRFs")
    
    
    parser.add_argument("--do_val", type=nice_str2bool,default=True, 
                    help="want to do model validation? 1 for yes, 0 for no")
    parser.add_argument("--do_varpart", type=nice_str2bool,default=False,
                    help="want to do variance partition? 1 for yes, 0 for no")
    parser.add_argument("--include_solo_models", type=nice_str2bool,default=True,
                    help="in varpart, want to fit each of the component models alone? 1 for yes, 0 for no")
    
    parser.add_argument("--do_tuning", type=nice_str2bool,default=False,
                    help="want to estimate tuning based on correlations? 1 for yes, 0 for no")
    parser.add_argument("--do_sem_disc", type=nice_str2bool,default=False,
                    help="want to estimate semantic discriminability? 1 for yes, 0 for no")
    parser.add_argument("--overwrite_sem_disc", type=nice_str2bool,default=False,
                    help="want to re-do (overwrite) semantic discriminability? 1 for yes, 0 for no")
    parser.add_argument("--date_str", type=str,default='',
                    help="what date was the model fitting done (only if you're starting from validation step.)")
    
     
    parser.add_argument("--sample_batch_size", type=int,default=500,
                    help="number of trials to analyze at once when making features (smaller will help with out-of-memory errors)")
    parser.add_argument("--voxel_batch_size", type=int,default=1000,
                    help="number of voxels to analyze at once when fitting weights (smaller will help with out-of-memory errors)")
    parser.add_argument("--voxel_batch_size_outer", type=int,default=1000,
                    help="number of voxels to analyze at once for permutation test (smaller will help with out-of-memory errors)")
    
   

    # Stuff that is specific to 'gabor' models
    parser.add_argument("--n_ori_gabor", type=int,default=4,
                    help="number of orientation channels to use")
    parser.add_argument("--n_sf_gabor", type=int,default=4,
                    help="number of spatial frequency channels to use")
    parser.add_argument("--gabor_nonlin_fn", type=nice_str2bool,default=True,
                    help="want to add nonlinearity to gabor features? 1 for yes, 0 for no")
    parser.add_argument("--use_pca_gabor_feats", type=nice_str2bool,default=False,
                    help="Want to use reduced dim (PCA) version of gabor features?")
    parser.add_argument("--use_fullimage_gabor_feats", type=nice_str2bool,default=False,
                    help="Want to use full-image (no pRFs) vers of gabor features?")
    
    # Specific to color models
    parser.add_argument("--use_fullimage_color_feats", type=nice_str2bool,default=False,
                    help="Want to use full-image (no pRFs) vers of color features?")
      
    # Specific to gist models
    parser.add_argument("--n_ori_gist", type=int,default=4,
                    help="number of orientation channels to use")
    parser.add_argument("--n_blocks_gist", type=int,default=4,
                    help="number of spatial grid blocks for gist model")
    
    # Stuff that is specific to pyramid model
    parser.add_argument("--n_ori_pyr", type=int,default=4,
                    help="number of orientation channels to use")
    parser.add_argument("--n_sf_pyr", type=int,default=4,
                    help="number of spatial frequency channels to use")
    parser.add_argument("--pyr_pca_type", type=str,default=None,
                    help="what pca type was used for texture features?")
    parser.add_argument("--group_all_hl_feats", type=nice_str2bool,default=True, 
                    help="want to simplify groups of features in texture model? 1 for yes, 0 for no")
    parser.add_argument("--do_pyr_varpart", type=nice_str2bool,default=False, 
                    help="want to do variance partition within texture model features? 1 for yes, 0 for no")
    
    # Specific to sketch tokens
    parser.add_argument("--use_pca_st_feats", type=nice_str2bool,default=False,
                    help="Want to use reduced dim (PCA) version of sketch tokens features?")
    parser.add_argument("--use_residual_st_feats", type=nice_str2bool,default=False,
                    help="Want to use sketch tokens features with gabor features regressed out?")
    parser.add_argument("--use_grayscale_st_feats", type=nice_str2bool,default=False,
                    help="Want to use sketch tokens features from grayscale images?")
    parser.add_argument("--use_fullimage_st_feats", type=nice_str2bool,default=False,
                    help="Want to use full-image (no pRFs) vers of sketchtokens features?")
    parser.add_argument("--st_pooling_size", type=int,default=4,
                    help="pooling kernel size used to compute full-image sketch tokens features")
    parser.add_argument("--st_use_avgpool", type=nice_str2bool,default=False,
                    help="For full-image sketchtokens features, use average pooling? (false=maxpool)")
    
    # specific to alexnet
    parser.add_argument("--alexnet_layer_name", type=str, default='', 
                       help="What layer of alexnet to use?")
    parser.add_argument("--alexnet_padding_mode", type=str, default='', 
                       help="What padding mode for alexnet conv layers? default zeros.")
    parser.add_argument("--use_pca_alexnet_feats", type=nice_str2bool,default=True, 
                       help="use reduced-dim version of alexnet features?")
    parser.add_argument("--alexnet_blurface", type=nice_str2bool,default=False, 
                       help="use version of alexnet features trained with blurry faces?")
    parser.add_argument("--use_fullimage_alexnet_feats", type=nice_str2bool,default=False,
                    help="Want to use full-image (no pRFs) vers of alexnet features?")
    
    # specific to CLIP/Resnet
    parser.add_argument("--resnet_layer_name", type=str, default='', 
                       help="What layer of resnet to use?")
    parser.add_argument("--resnet_model_architecture", type=str, default='RN50', 
                       help="What model architecture used for this version of resnet?")
    parser.add_argument("--resnet_training_type", type=str, default='', 
                       help="What training type used for resnet?")
    parser.add_argument("--use_pca_resnet_feats", type=nice_str2bool, default=True, 
                       help="use reduced-dim version of resnet features?")
    parser.add_argument("--use_fullimage_resnet_feats", type=nice_str2bool,default=False,
                    help="Want to use full-image (no pRFs) vers of resnet features?")
    
    parser.add_argument("--n_resnet_blocks_include", type=int,default=16,
                    help="when choosing best resnet layer, how many blocks to choose from? fewer will run faster")
    parser.add_argument("--resnet_blurface", type=nice_str2bool,default=False, 
                       help="use version of alexnet features trained with blurry faces?")

    # Specific to semantic models
    parser.add_argument("--semantic_feature_set", type=str,default='',
                    help="if semantic model, what dimension?")
    parser.add_argument("--use_fullimage_sem_feats", type=nice_str2bool,default=False,
                    help="Want to use full-image (no pRFs) vers of semantic features?")
    
    args = parser.parse_args()
    
    if args.prf_fixed_sigma==0:
        args.prf_fixed_sigma=None
    if args.pyr_pca_type=='None':
        args.pyr_pca_type = None
    if args.image_set=='none':
        args.image_set = None
        
    if args.shuffle_data:
        if args.shuff_rnd_seed==0:
            args.shuff_rnd_seed = int(time.strftime('%M%H%d', time.localtime()))
        print('\nPermutation test: random seed is %d\n'%args.shuff_rnd_seed)
    
    # print values of a few key things to the command line...
    if args.debug==1:
        print('USING DEBUG MODE...')
    
    return args
    