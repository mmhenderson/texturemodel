import argparse
import numpy as np
import sys, os
import torch
import time
import h5py
import torch.nn

#import custom modules
from utils import torch_utils,  default_paths
from utils import nsd_utils, floc_utils
from model_fitting import initialize_fitting
from feature_extraction import texture_statistics_pyramid

if torch.cuda.is_available():
    device = initialize_fitting.init_cuda()
else:
    device = 'cpu:0'

def extract_features(image_data, \
                     n_ori=4, n_sf=4, \
                     batch_size=100,
                     which_prf_grid=1, debug=False):
 
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid=which_prf_grid)    

    # Set up the pyramid    
    n_prf_sd_out = 2
    aperture=1.0
    _fmaps_fn = texture_statistics_pyramid.steerable_pyramid_extractor(pyr_height = n_sf, n_ori = n_ori)
    _feature_extractor = \
        texture_statistics_pyramid.texture_feature_extractor(_fmaps_fn,sample_batch_size=batch_size, \
                                n_prf_sd_out=n_prf_sd_out, aperture=aperture,device=device)

    n_features = _feature_extractor.n_features_total
    n_images = image_data.shape[0]
    n_prfs = models.shape[0]
    n_batches = int(np.ceil(n_images/batch_size))

    features_each_prf = np.zeros((n_images, n_features, n_prfs), dtype=np.float32)

    for bb in range(n_batches):

        if debug and bb>1:
            continue

        batch_inds = np.arange(batch_size * bb, np.min([batch_size * (bb+1), n_images]))

        print('Extracting features for images [%d - %d]'%(batch_inds[0], batch_inds[-1]))

        image_batch = torch_utils._to_torch(image_data[batch_inds,:,:,:], device)

#         image_batch = resample_fn(image_batch)

        _feature_extractor.clear_big_features()

        for mm in range(n_prfs):

            if debug and mm>1:
                continue

            x,y,sigma = models[mm,:]
            print('Getting features for pRF [x,y,sigma]:')
            print([x,y,sigma])

            features_batch = _feature_extractor(image_batch, models[mm],mm)

            print('model %d, min/max of features in batch: [%s, %s]'%(mm, torch.min(features_batch), torch.max(features_batch))) 

            features_each_prf[batch_inds,:,mm] = torch_utils.get_value(features_batch)

            sys.stdout.flush()

    return features_each_prf
    

def proc_one_subject(subject, args):
    
    if args.use_node_storage:
        pyramid_texture_feat_path = default_paths.pyramid_texture_feat_path_localnode
    else:
        pyramid_texture_feat_path = default_paths.pyramid_texture_feat_path
    if args.debug:
        pyramid_texture_feat_path = os.path.join(pyramid_texture_feat_path, 'DEBUG')
    if not os.path.exists(pyramid_texture_feat_path):
        os.makedirs(pyramid_texture_feat_path)
         
    filename_save = os.path.join(pyramid_texture_feat_path, \
                       'S%d_features_each_prf_%dori_%dsf_grid%d.h5py'%\
                         (subject, args.n_ori, args.n_sf, args.which_prf_grid))
            
    # Load and prepare the image set to work with (all images for the current subject, 10,000 ims)
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
      
    features_each_prf = extract_features(image_data, \
                                         n_ori=args.n_ori, n_sf=args.n_sf, \
                                         batch_size=args.batch_size,
                                         which_prf_grid=args.which_prf_grid, \
                                         debug=args.debug)

    save_features(features_each_prf, filename_save)
    

def proc_other_image_set(image_set, args):
    
    if args.use_node_storage:
        pyramid_texture_feat_path = default_paths.pyramid_texture_feat_path_localnode
    else:
        pyramid_texture_feat_path = default_paths.pyramid_texture_feat_path
    if args.debug:
        pyramid_texture_feat_path = os.path.join(pyramid_texture_feat_path, 'DEBUG')
    if not os.path.exists(pyramid_texture_feat_path):
        os.makedirs(pyramid_texture_feat_path)
       
    filename_save = os.path.join(pyramid_texture_feat_path, \
                       '%s_features_each_prf_%dori_%dsf_grid%d.h5py'%\
                         (image_set, args.n_ori, args.n_sf, args.which_prf_grid))
            
    if image_set=='floc':
        image_data = floc_utils.load_floc_images(npix=240)
    else:
        raise ValueError('image set %s not recognized'%image_set)
       
    features_each_prf = extract_features(image_data, \
                                         n_ori=args.n_ori, n_sf=args.n_sf, \
                                         batch_size=args.batch_size,
                                         which_prf_grid=args.which_prf_grid, \
                                         debug=args.debug)

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
                    help="how many frequency channels (pyramid levels)?")
    parser.add_argument("--batch_size", type=int,default=100,
                    help="batch size to use for feature extraction")
    
    parser.add_argument("--use_node_storage", type=int,default=0,
                    help="want to save and load from scratch dir on current node? 1 for yes, 0 for no")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which version of prf grid to use")
    
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
        
    