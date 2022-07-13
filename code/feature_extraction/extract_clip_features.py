import numpy as np
import sys, os
import argparse
import gc
import torch
import time
import h5py
import copy
import torch.nn as nn

#import custom modules
from utils import prf_utils, torch_utils, texture_utils, default_paths, nsd_utils
from model_fitting import initialize_fitting
from feature_extraction import pca_feats

# clip implemented in this package, from:
# https://github.com/openai/CLIP
import clip

# Define stuff about the resnet layers here
n_features_each_resnet_block = [256,256,256, 512,512,512,512, 1024,1024,1024,1024,1024,1024, 2048,2048,2048]
resnet_block_names = ['block%d'%nn for nn in range(len(n_features_each_resnet_block))]

def get_features_each_prf(subject, block_inds_do, use_node_storage=False, debug=False, \
                          which_prf_grid=1, save_dtype=np.float32):
    """
    Extract the portion of CNN feature maps corresponding to each pRF in the grid.
    Save values of the features in each pRF, for each layer of interest.
    """
    
    device = initialize_fitting.init_cuda()
    
    if use_node_storage:
        clip_feat_path = default_paths.clip_feat_path_localnode
    else:
        clip_feat_path = default_paths.clip_feat_path

    # Load and prepare the image set to work with (all images for the current subject, 10,000 ims)
    stim_root = default_paths.stim_root
    image_data = nsd_utils.get_image_data(subject)  
    image_data = nsd_utils.image_uncolorize_fn(image_data)
   
    n_images = image_data.shape[0]
    
    # Params for the spatial aspect of the model (possible pRFs)
    prf_models = initialize_fitting.get_prf_models(which_grid=which_prf_grid)    
    n_prfs = len(prf_models)
    
    prf_batch_size = 100;
    n_prf_batches = int(np.ceil(n_prfs/prf_batch_size))
    
    # Keep these params fixed
    n_prf_sd_out = 2
    batch_size = 100
    mult_patch_by_prf = True
    do_avg_pool = True
    model_architecture='RN50'

    n_blocks = len(resnet_block_names)
    
    n_batches = int(np.ceil(n_images/batch_size))

    with torch.no_grad():

        for ll in block_inds_do:
            
            block_inds = [ll]

            # separate the prfs into batches to save memory.  
            for pb in range(n_prf_batches):

                prfs_this_batch = np.arange(pb*prf_batch_size, np.min([(pb+1)*prf_batch_size, n_prfs]))
                print('pRF batch %d, includes:'%pb)
                print(prfs_this_batch)
                
                # doing the whole procedure of feature extraction one layer at a time
                # otherwise will run out of memory bc huge arrays.
                features_each_prf = np.zeros((n_images, n_features_each_resnet_block[ll], \
                                              len(prfs_this_batch)),dtype=save_dtype)
                
                for bb in range(n_batches):

                    if debug and bb>1:
                        continue
                    print('Processing images for batch %d of %d'%(bb, n_batches))

                    batch_inds = np.arange(batch_size * bb, np.min([batch_size * (bb+1), n_images]))

                    # using grayscale images for better comparison w my other models.
                    # need to tile to 3 so model weights will be right size
                    image_batch = np.tile(image_data[batch_inds,:,:,:], [1,3,1,1])

                    gc.collect()
                    torch.cuda.empty_cache()

                    activ_batch = get_clip_activations_batch(image_batch, block_inds, \
                                                         model_architecture, device=device)


                    print('Getting prf-specific activations for %s'%resnet_block_names[ll])

                    maps_full_field = torch.moveaxis(activ_batch[0], [0,1,2,3], [0,3,1,2])

                    if bb==0:
                        print('size of maps stack for first batch is:')
                        print(maps_full_field.shape)

                    for mi, mm in enumerate(prfs_this_batch):

                        if debug and mi>1:
                            continue

                        prf_params = prf_models[mm,:]
                        x,y,sigma = prf_params
                        print('Getting features for pRF [x,y,sigma]:')
                        print([x,y,sigma])
                        n_pix = maps_full_field.shape[1]

                        # Define the RF for this "model" version
                        prf = torch_utils._to_torch(prf_utils.gauss_2d(center=[x,y], sd=sigma, \
                                   patch_size=n_pix, aperture=1.0, dtype=np.float32), device=device)
                        minval = torch.min(prf)
                        maxval = torch.max(prf-minval)
                        prf_scaled = (prf - minval)/maxval

                        if mult_patch_by_prf:
                            # This effectively restricts the spatial location, so no need to crop
                            maps = maps_full_field * prf_scaled.view([1,n_pix, n_pix,1])
                        else:
                            # This is a coarser way of choosing which spatial region to look at
                            # Crop the patch +/- n SD away from center
                            bbox = texture_utils.get_bbox_from_prf(prf_params, prf.shape, n_prf_sd_out, \
                                                           min_pix=None, verbose=False, force_square=False)
                            print('bbox to crop is:')
                            print(bbox)
                            maps = maps_full_field[:,bbox[0]:bbox[1], bbox[2]:bbox[3],:]

                        if do_avg_pool:
                            features_batch = torch.mean(maps, dim=(1,2))
                        else:
                            features_batch = torch.max(maps, dim=(1,2))

                        print('model %d, min/max of features in batch: [%s, %s]'%(mm, \
                                                      torch.min(features_batch), torch.max(features_batch))) 

                        features_each_prf[batch_inds,:,mi] = torch_utils.get_value(features_batch)

                # Now save the results
                fn2save = os.path.join(clip_feat_path, \
                       'S%d_%s_%s_features_each_prf_grid%d_prfbatch%d.h5py'%(subject, model_architecture,\
                                               resnet_block_names[ll], which_prf_grid, pb))
                print('Writing prf features to %s\n'%fn2save)

                t = time.time()
                with h5py.File(fn2save, 'w') as data_set:
                    dset = data_set.create_dataset("features", np.shape(features_each_prf), dtype=save_dtype)
                    data_set['/features'][:,:,:] = features_each_prf
                    data_set.close() 
                elapsed = time.time() - t

                print('Took %.5f sec to write file'%elapsed)


def get_clip_activations_batch(image_batch, block_inds, model_architecture, device=None):

    """
    Get activations for images in NSD, passed through pretrained CLIP model.
    Specify which NSD images to look at, and which layers to return.
    """

    if device is None:
        device = torch.device('cpu:0')
       
    model, preprocess = clip.load(model_architecture, device=device)
    model.eval()
    
    # The 16 residual blocks are segmented into 4 groups here, which have different numbers of features.
    blocks_each= [len(model.visual.layer1), len(model.visual.layer2), len(model.visual.layer3),len(model.visual.layer4)]
    which_group = np.repeat(np.arange(4), blocks_each)

    activ = [[] for ll in block_inds]
    hooks = [[] for ll in block_inds]
    
    # first making this subfunction that is needed to get the activation on a forward pass
    def get_activ_fwd_hook(ii,ll):
        def hook(self, input, output):
            # the relu operation is used multiple times per block, but we only 
            # want to save its output when it has this specific size.
            if output.shape[1]==n_features_each_resnet_block[ll]:
                print('executing hook for %s'%resnet_block_names[ll])  
                activ[ii] = output
                print(output.shape)
        return hook

    image_tensors = torch.Tensor(image_batch).to(device)
    with torch.no_grad():

        # adding a "hook" to the module corresponding to each layer, so we'll save activations at each layer.
        # For resnet, going to save output of each residual block following last relu operation.
        for ii, ll in enumerate(block_inds):
            if which_group[ll]==0:            
                h = model.visual.layer1[ll].relu.register_forward_hook(get_activ_fwd_hook(ii,ll))
            elif which_group[ll]==1:            
                h = model.visual.layer2[ll-blocks_each[0]].relu.register_forward_hook(get_activ_fwd_hook(ii,ll))
            elif which_group[ll]==2:            
                h = model.visual.layer3[ll-sum(blocks_each[0:2])].relu.register_forward_hook(get_activ_fwd_hook(ii,ll))
            elif which_group[ll]==3:            
                h = model.visual.layer4[ll-sum(blocks_each[0:3])].relu.register_forward_hook(get_activ_fwd_hook(ii,ll))
            else:
                h=None
            hooks[ii] = h

        # Pass images though the model (hooks get run now)
        image_features = model.encode_image(image_tensors)

        # Now remove all the hooks
        for ii, ll in enumerate(block_inds):
            print(activ[ii].shape)
            hooks[ii].remove

    # Sanity check that we grabbed the right activations - check their sizes against expected
    # output size of each block
    exp_size = np.array(n_features_each_resnet_block)[block_inds]
    actual_size = [activ[bb].shape[1] for bb in range(len(activ))]
    assert(np.all(np.array(actual_size)==np.array(exp_size)))

    return activ


    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=1,
                    help="number of the subject, 1-8")
    parser.add_argument("--use_node_storage", type=int,default=0,
                    help="want to save and load from scratch dir on current node? 1 for yes, 0 for no")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which version of prf grid to use")
    parser.add_argument("--max_pc_to_retain", type=int,default=0,
                    help="max pc to retain? enter 0 for None")
    parser.add_argument("--min_pct_var", type=int,default=95,
                    help="min pct var to explain? default 95")
    
    args = parser.parse_args()
    
    n_blocks = len(resnet_block_names)  
    if args.subject==6:
        blocks_to_do = np.arange(13,16)
    else:
        blocks_to_do = np.arange(16)
                                
    for ll in blocks_to_do:

        # The clip activations are big, so going to make each layer and then delete
        # it as soon as pca is done.
        get_features_each_prf(subject = args.subject, block_inds_do = [ll], use_node_storage = args.use_node_storage, debug = args.debug==1, which_prf_grid=args.which_prf_grid)

        sys.stdout.flush()
            
        layer = 'block%d'%(ll)
        pca_feats.run_pca_clip(subject=args.subject, layer_name=layer, min_pct_var=args.min_pct_var, max_pc_to_retain=args.max_pc_to_retain, debug=args.debug==1, zscore_first=False, which_prf_grid=args.which_prf_grid)  
        
        model_architecture = 'RN50'
        if args.use_node_storage:
            clip_feat_path = default_paths.clip_feat_path_localnode
        else:
            clip_feat_path = default_paths.clip_feat_path
            
        n_prf_batches = 15;
        for pb in range(n_prf_batches):
            big_fn = os.path.join(clip_feat_path, \
                       'S%d_%s_%s_features_each_prf_grid%d_prfbatch%d.h5py'%(args.subject, model_architecture,\
                                               resnet_block_names[ll], args.which_prf_grid, pb))
            print('removing big activations file: %s'%big_fn)
            sys.stdout.flush()
            os.remove(big_fn)

            print('big file removed.')