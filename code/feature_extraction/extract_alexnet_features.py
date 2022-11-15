import numpy as np
import sys, os
import argparse
import gc
import torch
import time
import h5py
import copy
from collections import OrderedDict
import torchvision.models as models
import torch.nn as nn

#import custom modules
from utils import prf_utils, torch_utils, texture_utils, default_paths, nsd_utils, floc_utils
from model_fitting import initialize_fitting

dtype=np.float32

# Define sets of alexnet layers
alexnet_conv_layer_names = ['Conv1','Conv1_ReLU','Conv1_MaxPool', \
                       'Conv2','Conv2_ReLU','Conv2_MaxPool', \
                       'Conv3','Conv3_ReLU', \
                       'Conv4','Conv4_ReLU', \
                       'Conv5','Conv5_ReLU','Conv5_MaxPool']

alexnet_fc_layer_names = ['Dropout6','FC6','FC6_ReLU','Dropout7','FC7','FC7_ReLU','FC8']

n_feature_layers = len(alexnet_conv_layer_names)
n_classif_layers = len(alexnet_fc_layer_names)
n_total_layers = n_feature_layers + n_classif_layers
alexnet_layer_names = copy.deepcopy(alexnet_conv_layer_names)
alexnet_layer_names.extend(alexnet_fc_layer_names)

n_features_each_layer = [64,64,64, 192,192,192, 384,384, 256,256, 256,256]

try:
    device = initialize_fitting.init_cuda()
except:
    device = 'cpu:0'


def extract_features(image_data, layer_inds,\
                          padding_mode=None,\
                          which_prf_grid=5,\
                          batch_size=50, \
                          blurface=False, \
                          debug=False):

    """
    Extract the portion of CNN feature maps corresponding to pRF defined in "models"
    Return list of the features in each pRF, for each layer of interest.
    """

    if padding_mode=='':
        padding_mode = None
     
    n_images = image_data.shape[0]
    
    # Params for the spatial aspect of the model (possible pRFs)
    prf_models = initialize_fitting.get_prf_models(which_grid=which_prf_grid)    

    # Fix these params
    n_prf_sd_out = 2
    mult_patch_by_prf = True
    do_avg_pool = True
    
    n_layers = len(layer_inds)

    n_prfs = len(prf_models)
    features_each_prf = [np.zeros((n_images, n_features_each_layer[ll], n_prfs),dtype=dtype) for ll in layer_inds]

    n_batches = int(np.ceil(n_images/batch_size))

    with torch.no_grad():

        for bb in range(n_batches):

            if debug and bb>1:
                continue
            print('Processing images for batch %d of %d'%(bb, n_batches))

            batch_inds = np.arange(batch_size * bb, np.min([batch_size * (bb+1), n_images]))

            # using grayscale images for better comparison w my other models.
            # need to tile to 3 so alexnet weights will be right size
            image_batch = np.tile(image_data[batch_inds,:,:,:], [1,3,1,1])

            gc.collect()
            torch.cuda.empty_cache()
            
            activ_batch = get_alexnet_activations_batch(image_batch, layer_inds, \
                                                        device=device, \
                                                        padding_mode=padding_mode, \
                                                        blurface=blurface)

            for ll in range(n_layers):

                print('Getting prf-specific activations for layer %s'%alexnet_layer_names[layer_inds[ll]])

                maps_full_field = torch.moveaxis(activ_batch[ll], [0,1,2,3], [0,3,1,2])

                for mm in range(n_prfs):

                    if debug and mm>1:
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

                    if sigma==10:
                        # creating a "flat" pRF here which will average across entire feature map.
                        prf_scaled = torch.ones(prf_scaled.shape)
                        prf_scaled = prf_scaled/torch.sum(prf_scaled)
                        prf_scaled = prf_scaled.to(device)

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

                    features_each_prf[ll][batch_inds,:,mm] = torch_utils.get_value(features_batch)

    return features_each_prf
    

def get_alexnet_activations_batch(image_batch, layer_inds, device=None, padding_mode=None, blurface=False):

    """
    Get activations for images in NSD, passed through pretrained AlexNet.
    Specify which NSD images to look at, and which layers to return.
    """

    if device is None:
        device = torch.device('cpu:0')
       
    if blurface:
        # use model trained on face-blurred imagenet ims
        model_filename = os.path.join(default_paths.alexnet_blurface_feat_path, 'alexnet_blurred_ILSVRC.pth')
        print('Loading saved model from %s'%model_filename)
        saved = torch.load(model_filename, map_location=device)     
        sd = saved['state_dict']
        
        # need to fix keys in the statedict to have expected names
        new_sd = OrderedDict([])
        keynames = list(sd.keys())
        for kn in keynames:
            if 'features' in kn:
                new_kn = 'features.' + kn.split('features.module.')[1]
            else:
                new_kn = kn
            new_sd[new_kn] = sd[kn]
            
        model = models.alexnet().float().to(device)
        model.load_state_dict(new_sd)

    else:       
        # normal pre-trained model from torch model zoo
        print('Loading alexnet pre-trained on imagenet')
        model = models.alexnet(pretrained=True).float().to(device)
        
        
    if padding_mode is not None:
        # change padding type for all convolutional layers, "reflect" is a
        # good way to minimize edge artifacts.
        for ff in model.features:
            if hasattr(ff, 'padding_mode'):
                ff.padding_mode=padding_mode
                print('changing padding mode to %s'%padding_mode)
                print(ff)
                
                
    model.eval()
    
    is_fc = [('FC' in alexnet_layer_names[ll] or 'fc' in alexnet_layer_names[ll]) for ll in layer_inds]
    
    if len(layer_inds)==0:
        raise ValueError('your layer names do not match any of those specified in alexnet_features.py')

    
    # first making this subfunction that is needed to get the activation on a forward pass
    def get_activ_fwd_hook(ii,ll):
        def hook(self, input, output):            
            print('hook for %s'%alexnet_layer_names[ll])           
            activ[ii] = output
            print(output.shape)
        return hook
   
    # get image and labels for this batch
    # image_tensors is [batch_size x 3 x 224 x 224]
    image_tensors =  torch_utils._to_torch(image_batch, device=device).float()
    activ = [[] for ll in layer_inds]
    hook = [[] for ll in layer_inds]
    
    model.eval()

    # adding this "hook" to the module corresponding to each layer, so we'll save activations at each layer
    # this only modifies the "graph" e.g. what the model code does when run, but doesn't actually run it yet.
    for ii, ll in enumerate(layer_inds):
        if not is_fc[ii]:
            h = model.features[ll].register_forward_hook(get_activ_fwd_hook(ii,ll))
        else:
            h = model.classifier[ll-n_feature_layers].register_forward_hook(get_activ_fwd_hook(ii,ll))
        hook[ii] = h

    # do the forward pass of model, which now includes the forward hooks
    # now the "activ" variable will get modified, because it gets altered during the hook function
    model(image_tensors)
    
    # Now remove all the hooks
    for ii, ll in enumerate(layer_inds):
        print(activ[ii].shape)
        hook[ii].remove

    return activ




def proc_one_subject(subject, args):

    if args.blurface==1:
        feat_path = default_paths.alexnet_blurface_feat_path
    else:
        feat_path = default_paths.alexnet_feat_path
        
    if args.debug: 
        feat_path = os.path.join(feat_path, 'DEBUG') 
    if not os.path.exists(feat_path):
        os.makedirs(feat_path)

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
    
    layers_to_return = ['Conv1_ReLU', 'Conv2_ReLU','Conv3_ReLU','Conv4_ReLU','Conv5_ReLU']
    layer_inds = [ll for ll in range(len(alexnet_layer_names)) \
                      if alexnet_layer_names[ll] in layers_to_return]
    layer_inds = layer_inds[args.start_layer:]
    
    features_each_prf = extract_features(image_data, \
                                         layer_inds = layer_inds,
                                         padding_mode=args.padding_mode, \
                                         which_prf_grid=args.which_prf_grid, \
                                         batch_size=args.batch_size,
                                         blurface=args.blurface==1, 
                                         debug=args.debug)
    
    # Now save the results, one file for each alexnet layer 
    for ii, ll in enumerate(layer_inds):
        if args.padding_mode is not None:
            filename_save = os.path.join(feat_path, \
           'S%d_%s_%s_features_each_prf_grid%d.h5py'%(subject, \
                               alexnet_layer_names[ll], args.padding_mode, args.which_prf_grid))
        else:    
            filename_save = os.path.join(feat_path, \
               'S%d_%s_features_each_prf_grid%d.h5py'%(subject, \
                                       alexnet_layer_names[ll], args.which_prf_grid))
            
        save_features(features_each_prf[ii], filename_save)

    
def proc_other_image_set(image_set, args):
       
    if args.blurface==1:
        feat_path = default_paths.alexnet_blurface_feat_path
    else:
        feat_path = default_paths.alexnet_feat_path
      
    if args.debug: 
        feat_path = os.path.join(feat_path, 'DEBUG') 
    if not os.path.exists(feat_path):
        os.makedirs(feat_path)
      
    if image_set=='floc':
        image_data = floc_utils.load_floc_images(npix=240)
    else:
        raise ValueError('image set %s not recognized'%image_set)
       
    layers_to_return = ['Conv1_ReLU', 'Conv2_ReLU','Conv3_ReLU','Conv4_ReLU','Conv5_ReLU']
    layer_inds = [ll for ll in range(len(alexnet_layer_names)) \
                      if alexnet_layer_names[ll] in layers_to_return]
    layer_inds = layer_inds[args.start_layer:]
    
    features_each_prf = extract_features(image_data, \
                                         layer_inds = layer_inds,
                                         padding_mode=args.padding_mode, \
                                         which_prf_grid=args.which_prf_grid, \
                                         batch_size=args.batch_size,
                                         blurface=args.blurface==1, 
                                         debug=args.debug)
    
    # Now save the results, one file for each alexnet layer 
    for ii, ll in enumerate(layer_inds):
        if args.padding_mode is not None:
            filename_save = os.path.join(feat_path, \
           '%s_%s_%s_features_each_prf_grid%d.h5py'%(image_set, \
                               alexnet_layer_names[ll], args.padding_mode, args.which_prf_grid))
        else:    
            filename_save = os.path.join(feat_path, \
               '%s_%s_features_each_prf_grid%d.h5py'%(image_set, \
                                       alexnet_layer_names[ll], args.which_prf_grid))
            
        save_features(features_each_prf[ii], filename_save)

        
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
    parser.add_argument("--start_layer", type=int,default=0,
                    help="which network layer to start from?")
    parser.add_argument("--batch_size", type=int,default=50,
                    help="batch size")
    parser.add_argument("--debug", type=int,default=0,
                    help="want to run a fast test version of this script to debug? 1 for yes, 0 for no")
    parser.add_argument("--which_prf_grid", type=int,default=1,
                    help="which version of prf grid to use")
    parser.add_argument("--padding_mode", type=str,default='',
                    help="padding mode for alexnet convolutional layers")
    parser.add_argument("--blurface", type=int, default=0, 
                    help="use model trained with faces blurred? 1 for yes, 0 for no")
    
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
        
