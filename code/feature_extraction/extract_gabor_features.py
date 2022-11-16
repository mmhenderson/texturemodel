import argparse
import numpy as np
import sys, os
import torch
import time
import h5py

#import custom modules
from utils import numpy_utils, torch_utils, nsd_utils, prf_utils, default_paths, floc_utils
from model_fitting import initialize_fitting
from feature_extraction import gabor_feature_extractor

try:
    device = initialize_fitting.init_cuda()
except:
    device = 'cpu:0'
    
def extract_features(image_data,\
                     n_ori=4, n_sf=4, \
                     sample_batch_size=100, \
                     nonlin_fn=False, \
                     which_prf_grid=5, debug=False):
    
    """ Extract Gabor features for the images in image_data, 
        within each pRF of a specified grid
    
    Input: 
    image_data, size [n_images x n_color_channels x n_pix x n_pix]
    
    Returns:   
    features_each_prf, size [n_images x n_features x n_prfs]   
    
    """
 
    n_pix = image_data.shape[2]
    n_images = image_data.shape[0]
    n_batches = int(np.ceil(n_images/sample_batch_size))

    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid = which_prf_grid)    
    n_prfs = models.shape[0]
    
    # Set up the feature extractor module here
    if nonlin_fn:
        # adding a nonlinearity to the filter activations
        print('\nAdding log(1+sqrt(x)) as nonlinearity fn...')
        nonlin = lambda x: torch.log(1+torch.sqrt(x))
    else:
        nonlin = None
        
    _gabor_ext_complex = gabor_feature_extractor.gabor_extractor_multi_scale(n_ori=n_ori, n_sf=n_sf, \
                             sf_range_cyc_per_stim = (3, 72), log_spacing = True, \
                             pix_per_cycle=4.13, cycles_per_radius=0.7, radii_per_filter=4, \
                             complex_cell=True, padding_mode = 'circular', nonlin_fn=nonlin, \
                             RGB=False, device = device)
    
    _gabor_ext_complex.get_fmaps_sizes([n_pix, n_pix])   
    n_features = _gabor_ext_complex.n_features
    
    print('number of features total: %d'%n_features)
    
    features_each_prf = np.zeros((n_images, n_features, n_prfs), dtype=np.float32)

    with torch.no_grad():
        
        for bb in range(n_batches):

            if debug and bb>1:
                continue

            batch_inds = np.arange(sample_batch_size * bb, np.min([sample_batch_size * (bb+1), n_images]))

            print('Extracting features for images [%d - %d]'%(batch_inds[0], batch_inds[-1]))

            image_batch = torch_utils._to_torch(image_data[batch_inds,:,:,:], device)

            for mm in range(n_prfs):

                if debug and mm>1:
                    continue

                x,y,sigma = models[mm,:]
                print('Getting features for pRF [x,y,sigma]:')
                print([x,y,sigma])
                
                print('Computing complex cell features...')
                t = time.time()
                features_batch = get_avg_features_in_prf(_gabor_ext_complex, image_batch, models[mm,:],\
                                                                sample_batch_size=sample_batch_size, \
                                                                aperture=1.0, device=device, to_numpy=True)
                elapsed =  time.time() - t
                print('time elapsed = %.5f'%elapsed)

                print('model %d, min/max of features in batch: [%s, %s]'%(mm, \
                                          np.min(features_batch), np.max(features_batch))) 

                features_each_prf[batch_inds,:,mm] = features_batch

                sys.stdout.flush()
                
    return features_each_prf
                                              
    
    
def get_avg_features_in_prf(_fmaps_fn, images, prf_params, sample_batch_size, aperture, device, \
                            dtype=np.float32, to_numpy=True):
    
    """
    For a given set of images and a specified pRF position and size, compute the mean (weighted by pRF)
    in each feature map channel. Returns [nImages x nFeatures]
    """
    
    x,y,sigma = prf_params
    n_trials = images.shape[0]
    n_features = _fmaps_fn.n_features
    fmaps_rez = _fmaps_fn.resolutions_each_sf

    features = np.zeros(shape=(n_trials, n_features), dtype=dtype)
    if to_numpy==False:
         features = torch_utils._to_torch(features, device=device)

    # Define the RF for this "model" version - at several resolutions.
    _prfs = [torch_utils._to_torch(prf_utils.gauss_2d(center=[x,y], sd=sigma, \
                                   patch_size=n_pix, aperture=aperture, dtype=dtype), \
                                   device=device) for n_pix in fmaps_rez]

    # To make full design matrix for all trials, first looping over trials in batches to get the features
    # Only reason to loop is memory constraints, because all trials is big matrices.
    t = time.time()
    n_batches = np.ceil(n_trials/sample_batch_size)
    bb=-1
    for rt,rl in numpy_utils.iterate_range(0, n_trials, sample_batch_size):

        bb=bb+1

        # Multiplying feature maps by RFs here. 
        # Feature maps in _fm go [nTrials x nFeatures(orientations) x nPixels x nPixels]
        # Spatial RFs in _prfs go [nPixels x nPixels]
        # Once we multiply, get [nTrials x nFeatures]
        # note this is concatenating SFs together from low (smallest maps) to high (biggest maps). 
        # Cycles through all orient channels in order for first SF, then again for next SF, etc.
        _features = torch.cat([torch.tensordot(_fm, _prf, dims=[[2,3], [0,1]]) \
                               for _fm,_prf in zip(_fmaps_fn(images[rt]), _prfs)], dim=1) # [#samples, #features]

        # Add features for this batch to full design matrix over all trials
        if to_numpy:
            features[rt] = torch_utils.get_value(_features)
        else:
            features[rt] = _features

        elapsed = time.time() - t

    return features


def proc_one_subject(subject, args):

    if args.use_node_storage:
        gabor_texture_feat_path = default_paths.gabor_texture_feat_path_localnode
    else:
        gabor_texture_feat_path = default_paths.gabor_texture_feat_path
       
    if args.debug:
        gabor_texture_feat_path = os.path.join(gabor_texture_feat_path,'DEBUG')
       
    if not os.path.exists(gabor_texture_feat_path):
        os.makedirs(gabor_texture_feat_path)
     
    # Load and prepare the image set to work with 
    if subject==999:
        # 999 is a code i am using to indicate the independent set of coco images, which were
        # not actually shown to any NSD participants
        from utils import coco_utils
        image_data = coco_utils.load_indep_coco_images(n_pix=240)
        image_data = nsd_utils.image_uncolorize_fn(image_data)
    else: 
        # load all images for the current subject, 10,000 ims
        image_data = nsd_utils.get_image_data(subject)  
        image_data = nsd_utils.image_uncolorize_fn(image_data)

    filename_save = os.path.join(gabor_texture_feat_path, \
                               'S%d_features_each_prf_%dori_%dsf_gabor_solo'%\
                                 (subject, args.n_ori, args.n_sf))
    if args.nonlin_fn:
        filename_save += '_nonlin'
        
    filename_save += '_grid%d.h5py'%args.which_prf_grid   
        
    features_each_prf = extract_features(image_data, \
                                         n_ori=args.n_ori, n_sf=args.n_sf, \
                                         sample_batch_size=args.sample_batch_size, \
                                         nonlin_fn=args.nonlin_fn, \
                                         which_prf_grid=args.which_prf_grid, debug=args.debug)
    
    save_features(features_each_prf, filename_save)

    
def proc_other_image_set(image_set, args):
       
    if args.use_node_storage:
        gabor_texture_feat_path = default_paths.gabor_texture_feat_path_localnode
    else:
        gabor_texture_feat_path = default_paths.gabor_texture_feat_path
    
    if args.debug:
        gabor_texture_feat_path = os.path.join(gabor_texture_feat_path,'DEBUG')
        
    if not os.path.exists(gabor_texture_feat_path):
        os.makedirs(gabor_texture_feat_path)
        
    if image_set=='floc':
        image_data = floc_utils.load_floc_images(npix=240)
    else:
        raise ValueError('image set %s not recognized'%image_set)
        
    filename_save = os.path.join(gabor_texture_feat_path, \
                       '%s_features_each_prf_%dori_%dsf_gabor_solo'%(image_set, args.n_ori, args.n_sf))

    if args.nonlin_fn:
        filename_save += '_nonlin'
        
    filename_save += '_grid%d.h5py'%args.which_prf_grid   
        
    features_each_prf = extract_features(image_data, \
                                         n_ori=args.n_ori, n_sf=args.n_sf, \
                                         sample_batch_size=args.sample_batch_size, \
                                         nonlin_fn=args.nonlin_fn, \
                                         which_prf_grid=args.which_prf_grid, debug=args.debug)
    
    save_features(features_each_prf, filename_save)
                                    
    
def save_features(features_each_prf, filename_save):
    
    print('Writing prf features to %s\n'%filename_save)
    
    t = time.time()
    with h5py.File(filename_save, 'w') as data_set:
        dset = data_set.create_dataset("features", np.shape(features_each_prf), dtype=np.float32)
        data_set['/features'][:,:,:] = features_each_prf
        data_set.close()  
    elapsed = time.time() - t
    
    print('Took %.5f sec to write file'%elapsed)
    

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--subject", type=int,default=0,
                    help="number of the subject, 1-8")
    parser.add_argument("--image_set", type=str,default='none',
                    help="name of the image set to use (if not an NSD subject)")
    parser.add_argument("--n_ori", type=int,default=4,
                    help="how many orientation channels?")
    parser.add_argument("--n_sf", type=int,default=4,
                    help="how many frequency channels?")
    parser.add_argument("--sample_batch_size", type=int,default=100,
                    help="batch size to use for feature extraction")
    parser.add_argument("--nonlin_fn", type=int,default=0,
                    help="want to add nonlinearity to gabor features? 1 for yes, 0 for no")
    
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which version of prf grid to use")
    
    parser.add_argument("--use_node_storage", type=int,default=0,
                    help="want to save and load from scratch dir on current node? 1 for yes, 0 for no")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    
    args = parser.parse_args()
    
    if args.subject==0:
        args.subject=None
    if args.image_set=='none':
        args.image_set=None
        
    args.debug = (args.debug==1)     
    
    if args.subject is not None:
        
        proc_one_subject(subject = args.subject, args=args)
        
    elif args.image_set is not None:
        
        proc_other_image_set(image_set=args.image_set, args=args)
        
    