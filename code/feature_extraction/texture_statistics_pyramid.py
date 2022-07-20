import numpy as np
import torch
import time
import torch.nn as nn
import pyrtools as pt
from utils import numpy_utils, torch_utils, texture_utils, prf_utils, default_paths
from feature_extraction import texture_feature_utils

feature_types_all = texture_feature_utils.feature_type_names_raw
feature_type_dims_all = texture_feature_utils.feature_type_dims_raw

class texture_feature_extractor(nn.Module):
    
    """
    Module to compute higher-order texture statistics of input images 
    (e.g. Portilla & Simoncelli 2000, IJCV)
    Statistics are computed within a specified region of space (a voxel's pRF)
    Can specify different subsets of features to include (i.e. pixel-level stats, simple/complex cells, 
    cross-correlations, auto-correlations)
    Inputs to the forward pass are images and pRF parameters of interest [x,y,sigma]
    "fmaps_fn" is a steerable pyramid extractor object (see below)
    """
    
    def __init__(self,_fmaps_fn, subject=None, \
                 sample_batch_size=100, n_prf_sd_out=2, aperture=1.0, \
                 device=None):
        
        super(texture_feature_extractor, self).__init__()
        
        self.subject = subject
        self.fmaps_fn = _fmaps_fn   
        self.n_sf = _fmaps_fn.pyr_height
        self.n_ori =  _fmaps_fn.n_ori
       
        self.sample_batch_size = sample_batch_size       
        self.n_prf_sd_out = n_prf_sd_out
        self.aperture = aperture
        self.device = device       
        
        self.feature_types_include = feature_types_all
        self.feature_type_dims = feature_type_dims_all
        self.n_features_total = np.sum(self.feature_type_dims)
        
        self.fmaps = None
   
    def get_maps(self, images):
    
        print('Running steerable pyramid feature extraction...')
        print('Images array shape is:')
        print(images.shape)
        t = time.time()
        if isinstance(images, torch.Tensor):
            images = torch_utils.get_value(images)
        fmaps = self.fmaps_fn(images, to_torch=False, device=self.device)        
        self.fmaps = fmaps
        elapsed =  time.time() - t
        print('time elapsed = %.5f'%elapsed)

    def clear_big_features(self):
        
        print('Clearing steerable pyramid features from memory.')
        self.fmaps = None
        
    def forward(self, images, prf_params, prf_model_index):

        if self.fmaps is None:
            self.get_maps(images)
        else:
            assert(images.shape[0]==self.fmaps[0][0].shape[0])

        if isinstance(prf_params, torch.Tensor):
            prf_params = torch_utils.get_value(prf_params)
        assert(np.size(prf_params)==3)
        prf_params = np.squeeze(prf_params)
        if isinstance(images, torch.Tensor):
            images = torch_utils.get_value(images)

        print('Computing higher order correlations...')

        t = time.time()
        pars = get_all_features(self.fmaps, images, prf_params, \
                                sample_batch_size=self.sample_batch_size, \
                                n_prf_sd_out=self.n_prf_sd_out, \
                                aperture=self.aperture, device=self.device)
        all_feat = dict(zip(feature_types_all, pars))
        
        elapsed =  time.time() - t
        print('time elapsed = %.5f'%elapsed)

        # Now concatenating everything to a big matrix
        for ff, feature_name in enumerate(self.feature_types_include):   
            assert(all_feat[feature_name] is not None)
            assert(all_feat[feature_name].shape[1]==self.feature_type_dims[ff])
            if ff==0:
                all_feat_concat = all_feat[feature_name]
            else:               
                all_feat_concat = torch.cat((all_feat_concat, all_feat[feature_name]), axis=1)

        assert(all_feat_concat.shape[1]==self.n_features_total)
        print('Final size of features concatenated is [%d x %d]'%(all_feat_concat.shape[0], \
                                                                  all_feat_concat.shape[1]))
        print('Feature types included are:')
        print(self.feature_types_include)
        
        if torch.any(torch.isnan(all_feat_concat)):
            print('\nWARNING THERE ARE NANS IN FEATURES MATRIX\n')
        if torch.any(torch.all(all_feat_concat==0, axis=0)):
            print('\nWARNING THERE ARE ZEROS IN FEATURES MATRIX\n')
            print('zeros for columns:')
            print(np.where(torch_utils.get_value(torch.all(all_feat_concat==0, axis=0))))
                             
        print('Final size of features concatenated is [%d x %d]'%(all_feat_concat.shape[0], \
                                                                  all_feat_concat.shape[1]))
       
        return all_feat_concat
    
    
class steerable_pyramid_extractor(nn.Module):
    
    """
    Module that utilizes steerable pyramid ( https://pyrtools.readthedocs.io/en/latest/) 
    to extract features.
    For a batch of input images, will return all the pyramid coefficients, as well as 
    additional types of feature maps (i.e. partially reconstructed lowpass images at several 
    frequency levels, upsampled feature maps).
    These are used by 'get_all_features' to extract various textural features of the image.
    Adapted by MH from code in the library at:
    https://github.com/freeman-lab/metamers
    """
    
    def __init__(self, pyr_height=4, n_ori=8):
        
        super(steerable_pyramid_extractor, self).__init__()       
        self.n_ori = n_ori
        self.pyr_height = pyr_height # how many spatial frequencies?
        self.pyr = None
        
    def forward(self, image_batch, to_torch=False, device=None):
        
        batch_size = image_batch.shape[0]
        t  = time.time()
        for ii in range(batch_size):
            
            # Call the pyramid generation code here, get all features for this image.
            pyr = pt.pyramids.SteerablePyramidFreq(image_batch[ii,0,:,:], is_complex=True, \
                                                   height = self.pyr_height, order = self.n_ori-1)
            self.pyr = pyr # storing the most recently generated pyramid, in case we need its properties later
            
            if ii==0:  
                
                # Initialize all the feature maps we want to store...
                fmaps_complex = []
                fmaps_coarser_upsampled = []
               
                # Will generate several low-pass filtered representations of the image - 
                # will use these as additional feature maps for 
                # computing autocorrelations and marginal statistics.
                fmaps_lowpass_recon = []
                fmaps_lowpass_recon.append(np.zeros((batch_size,1,pyr.pyr_coeffs['residual_lowpass'].shape[0],\
                                                     pyr.pyr_coeffs['residual_lowpass'].shape[1])))
               
                # Feature maps will be listed from low SF to high SF. 
                # Each map stack is size [batch_size x orientations x height x width]
                # need to go backward because pyr comes out in the opposite order from what i want
                sf_reverse  = self.pyr_height  
                for sf in range(self.pyr_height):
                    sf_reverse -= 1
                    fmaps_complex.append(np.zeros((batch_size, self.n_ori,pyr.pyr_coeffs[sf_reverse,0].shape[0],\
                                                   pyr.pyr_coeffs[sf_reverse,0].shape[1]), dtype=complex))
                    
                    # Initialize "parent" representations for this level 
                    # (map from the next coarsest resolution, upsampled to the finer scale)
                    # To be used for cross-scale comparisons.
                    if sf==0:
                        # this will be the lowpass residual (non-oriented).
                        fmaps_coarser_upsampled.append(np.zeros((batch_size, 1, \
                                                                 pyr.pyr_coeffs[sf_reverse,0].shape[0],\
                                                                 pyr.pyr_coeffs[sf_reverse,0].shape[1]), \
                                                                 dtype=complex))                     
                    else:
                        fmaps_coarser_upsampled.append(np.zeros((batch_size, self.n_ori, \
                                                                 pyr.pyr_coeffs[sf_reverse,0].shape[0],\
                                                                 pyr.pyr_coeffs[sf_reverse,0].shape[1]), \
                                                                 dtype=complex))     
                        
                    fmaps_lowpass_recon.append(np.zeros((batch_size,1,pyr.pyr_coeffs[sf_reverse,0].shape[0],\
                                                         pyr.pyr_coeffs[sf_reverse,0].shape[1])))

                fmaps_resid = []    
                fmaps_resid.append(np.zeros((batch_size,1,pyr.pyr_coeffs['residual_lowpass'].shape[0],\
                                             pyr.pyr_coeffs['residual_lowpass'].shape[1])))
                fmaps_resid.append(np.zeros((batch_size,1,pyr.pyr_coeffs['residual_highpass'].shape[0],\
                                             pyr.pyr_coeffs['residual_highpass'].shape[1])))

            # First get lowpass filtered representation of the image
            lowpass_recon = pyr.recon_pyr(levels='residual_lowpass', bands='all', twidth=1) 
            scale_by = pyr.pyr_size[(0,0)][0]/pyr.pyr_size['residual_lowpass'][0]
            lowpass_recon = texture_utils.shrink(lowpass_recon, scale_by)*scale_by**2
            fmaps_lowpass_recon[0][ii,0,:,:] = lowpass_recon
            
            # Get the "parent" for lowest SF level (upsample the residual lowpass)
            upsampled = texture_utils.expand(pyr.pyr_coeffs['residual_lowpass'], factor=2)/2**2
            fmaps_coarser_upsampled[0][ii,0,:,:] = upsampled
            
            # Feature maps will be listed from low SF to high SF. 
            # Each map stack is size [batch_size x orientations x height x width]
            # need to go backward because pyr comes out in the opposite order from what i want
            sf_reverse  = self.pyr_height 
            for sf in range(self.pyr_height):
                sf_reverse -= 1
                for oo in range(self.n_ori):     

                    # These are the main feature maps of the pyramid - 
                    # one feature map per scale per orientation band.
                    # Complex number, can take the magnitude or real/imaginary part 
                    # to simulate complex or simple cell-type responses.
                    fmaps_complex[sf][ii,oo,:,:] = pyr.pyr_coeffs[(sf_reverse,oo)]
                    
                    if sf<self.pyr_height-1:
                        # Store this as a "parent" representation, 
                        # will be used for the next most fine SF level (i.e. sf+1)
                        upsampled = texture_utils.expand(pyr.pyr_coeffs[(sf_reverse,oo)], factor=2)/2**2
                        # Double the phase (angle of the complex number); note this doesn't affect the magnitude. 
                        phase_doubled = texture_utils.double_phase(upsampled)
                        fmaps_coarser_upsampled[sf+1][ii,oo,:,:] = phase_doubled
                   
                        
                # Get the bandpass filtered representation for this scale
                bandpass_image = np.real(pyr.recon_pyr(levels=sf_reverse, bands='all', twidth=1))
                scale_by = pyr.pyr_size[(0,0)][0]/pyr.pyr_size[(sf_reverse,0)][0]
                bandpass_image = texture_utils.shrink(bandpass_image, factor=scale_by)*scale_by**2
                
                # Add it onto the lowpass_recon (gets modified every loop iteration)
                lowpass_recon = texture_utils.expand(lowpass_recon, factor=2)/2**2
                lowpass_recon = lowpass_recon + bandpass_image
                fmaps_lowpass_recon[sf+1][ii,0,:,:] = lowpass_recon
            

            # Grab residual feature maps, the lowest and highest levels of the pyramid
            fmaps_resid[0][ii,0,:,:] = pyr.pyr_coeffs['residual_lowpass']
            fmaps_resid[1][ii,0,:,:] = pyr.pyr_coeffs['residual_highpass']
                        
        elapsed = time.time() - t
#         print('time elapsed: %.5f s'%elapsed)

        if to_torch:            
            fmaps_complex = [torch.from_numpy(fm).to(device) for fm in fmaps_complex]            
            fmaps_resid = [torch_utils._to_torch(fm, device=device) for fm in fmaps_resid]
            fmaps_lowpass_recon = [torch_utils._to_torch(fm, device=device) for fm in fmaps_lowpass_recon]                      
            fmaps_coarser_upsampled = [torch.from_numpy(fm).to(device) for fm in fmaps_coarser_upsampled]

        return fmaps_complex, fmaps_resid, fmaps_lowpass_recon, fmaps_coarser_upsampled
    
def get_all_features(fmaps, images, prf_params, \
                              sample_batch_size=20, n_prf_sd_out=2, \
                              aperture=1.0, device=None, \
                              keep_orig_shape=False):

    """
    Compute texture statistics features for a batch of images and a desired pRF.
    Input "fmaps" is the module that defines steerable pyramid 
    (i.e. 'steerable_pyramid_extractor').
    
    Returns arrays of each texture statistics feature.  
    Adapted by MH from code in the library at:
    https://github.com/freeman-lab/metamers
    """

    fmaps_complex_all, fmaps_resid_all, fmaps_lowpass_recon_all, fmaps_coarser_upsampled_all = fmaps
   
    n_trials = fmaps_complex_all[0].shape[0]
    x,y,sigma = prf_params

    n_sf = len(fmaps_complex_all)
    n_ori = fmaps_complex_all[0].shape[1]
        
    # all pairs of different orientation channels.
    ori_pairs = np.vstack([[[oo1, oo2] for oo2 in np.arange(oo1+1, n_ori)] \
                           for oo1 in range(n_ori) if oo1<n_ori-1])
    n_ori_pairs = np.shape(ori_pairs)[0]

    # mean, variance, skew, kurtosis, min, max
    pixel_stats = torch.zeros((n_trials,6), device=device)

    # Mean magnitude each scale/orientation, within the prf.
    mean_magnitudes = torch.zeros((n_trials, n_sf, n_ori), device=device)
    mean_realparts = torch.zeros((n_trials, n_sf, n_ori), device=device)

    # Store the skew and kurtosis of the lowpass reconstructions at each scale
    marginal_stats_lowpass_recons = torch.zeros((n_trials, n_sf+1, 2), device=device)

    # Variance of the highpass residual
    variance_highpass_resid = torch.zeros((n_trials, 1), device=device)

    # how many unique autocorrelation values will we get out for each feature map? 
    # These will be pre-defined, same for every pRF.
    # but different for different scales of feature maps.
    # note that for bigger prfs, there will potentially be more pixels that contribute 
    # to the autocorrelation computation (due to weights)- 
    # but a fixed portion of the matrix is returned.
    autocorr_output_pix=np.array([3,3,5,7,7])
    n_autocorr_vals = ((autocorr_output_pix**2+1)/2).astype('int')
    max_autocorr_vals = np.max(n_autocorr_vals)
    
    # Spatial autocorrelation of the magnitude of spectral coefficients, within each scale and orientation.
    magnitude_feature_autocorrs = torch.zeros([n_trials, n_sf, n_ori, max_autocorr_vals], \
                                              device=device) # this is ace in the matlab code

    # Spatial autocorrelation of the partially-reconstructed lowpass image representation at each scale
    lowpass_recon_autocorrs = torch.zeros([n_trials, n_sf+1, max_autocorr_vals], \
                                          device=device) # this is acr in the matlab code

    # Spatial autocorrelation of the highpass residual
    highpass_resid_autocorrs = torch.zeros([n_trials, 1, max_autocorr_vals], \
                                           device=device) # this is acr in the matlab code

    # Within scale correlations of feature maps: compare feature map magnitudes for different orientations.
    magnitude_within_scale_crosscorrs = torch.zeros([n_trials, n_sf, n_ori_pairs], \
                                                    device=device) # this is C0 in the matlab code
    # Using the real parts.
    real_within_scale_crosscorrs = torch.zeros([n_trials, n_sf, n_ori_pairs], \
                                               device=device) # this is Cr0 in the matlab code

    # Cross-scale correlations of feature maps
    # always comparing each scale to an up-sampled version of the scale coarser than it.
    magnitude_across_scale_crosscorrs = torch.zeros([n_trials, n_sf-1, n_ori, n_ori], \
                                                    device=device) # this is Cx0 in the matlab code

    # Cross-scale correlations, using the real and imaginary parts separately. 
    # The phase (angle) of the coarser map is doubled before computing these.
    real_imag_across_scale_crosscorrs = torch.zeros([n_trials, n_sf-1, 2, n_ori, n_ori], \
                                                    device=device) # this is Crx0 in the matlab code

    # These are comparisons with spatially shifted versions of the lowpass residual.
    # these are included in Cr0 and Crx0 in the matlab code
    n_spatshifts = 5;
    real_spatshift_within_scale_crosscorrs = torch.zeros([n_trials, 1, n_spatshifts, n_spatshifts], device=device)
    real_spatshift_across_scale_crosscorrs = torch.zeros([n_trials, 1, n_ori, n_spatshifts], device=device) 

    # Looping over batches of trials to compute everything of interest.
    bb=-1
    for batch_inds, batch_size_actual in numpy_utils.iterate_range(0, n_trials, sample_batch_size):
        bb=bb+1

        fmaps_complex = [torch.from_numpy(fmaps_complex_all[ii][batch_inds,:,:,:]).to(device) \
                                   for ii in range(len(fmaps_complex_all))]
        fmaps_resid = [torch.from_numpy(fmaps_resid_all[ii][batch_inds,:,:,:]).float().to(device) \
                                   for ii in range(len(fmaps_resid_all))]
        fmaps_lowpass_recon = [torch.from_numpy(fmaps_lowpass_recon_all[ii][batch_inds,:,:,:]).float().to(device) \
                                   for ii in range(len(fmaps_lowpass_recon_all))]
        fmaps_coarser_upsampled = [torch.from_numpy(fmaps_coarser_upsampled_all[ii][batch_inds,:,:,:]).to(device) \
                                   for ii in range(len(fmaps_coarser_upsampled_all))]

        if bb==0:
            npix_each_scale = [fmaps_complex_all[sc].shape[2] for sc in np.arange(n_sf-1,-1,-1)]
            npix_each_scale.append(fmaps_resid_all[0].shape[2])
            npix_each_scale.reverse()

        # First working with the finest scale (original image)
        n_pix = npix_each_scale[-1]   
        spatial_weights = prf_utils.gauss_2d(center=[x,y], sd=sigma, \
                                   patch_size=n_pix, aperture=aperture, dtype=np.float32)
        patch_bbox_square = texture_utils.get_bbox_from_prf(prf_params, spatial_weights.shape, \
                                                            n_prf_sd_out, force_square=True, \
                                                            min_pix=autocorr_output_pix[-1])

        # Gather pixel-wise statistics here 
        wmean, wvar, wskew, wkurt = texture_utils.get_weighted_pixel_features(images[batch_inds], \
                                                                              spatial_weights, device=device)
        pixel_stats[batch_inds,0] = torch.squeeze(wmean)
        pixel_stats[batch_inds,1] = torch.squeeze(wvar)
        pixel_stats[batch_inds,2] = torch.squeeze(wskew)
        pixel_stats[batch_inds,3] = torch.squeeze(wkurt)
        pixel_stats[batch_inds,4] = \
            torch_utils._to_torch(np.squeeze(np.min(np.min(images[batch_inds], \
                                                           axis=3), axis=2)), device=device)
        pixel_stats[batch_inds,5] = \
            torch_utils._to_torch(np.squeeze(np.max(np.max(images[batch_inds], \
                                                           axis=3), axis=2)), device=device)

        # Autocorrs of the highpass residual
        highpass_resid = fmaps_resid[1]
        auto_corr = texture_utils.weighted_auto_corr_2d(highpass_resid, spatial_weights, \
                                                        patch_bbox=patch_bbox_square, \
                                                        output_pix = autocorr_output_pix[-1], \
                                                        subtract_patch_mean = True, \
                                                        enforce_size=True, device=device)       
        highpass_resid_autocorrs[batch_inds,0,0:n_autocorr_vals[-1]] = \
            torch.reshape(texture_utils.unique_autocorrs(auto_corr),[batch_size_actual, n_autocorr_vals[-1]])

        # Variance of the highpass residual
        m, wvar, s, k = texture_utils.get_weighted_pixel_features(highpass_resid, spatial_weights, device=device)
        variance_highpass_resid[batch_inds,0] = torch.squeeze(wvar)

        # Next work with the low-pass reconstruction (most coarse scale, smallest npix)
        n_pix = npix_each_scale[0]              
        spatial_weights = prf_utils.gauss_2d(center=[x,y], sd=sigma, \
                                   patch_size=n_pix, aperture=aperture, dtype=np.float32)
        patch_bbox_square = texture_utils.get_bbox_from_prf(prf_params, spatial_weights.shape, \
                                                            n_prf_sd_out, force_square=True, \
                                                            min_pix=autocorr_output_pix[0])

        lowpass_rec = fmaps_lowpass_recon[0]

        # Marginal stats of low-pass reconstruction
        m, v, wskew, wkurt = texture_utils.get_weighted_pixel_features(lowpass_rec, \
                                                                       spatial_weights, \
                                                                       device=device)
        marginal_stats_lowpass_recons[batch_inds,0,0] = torch.squeeze(wskew)
        marginal_stats_lowpass_recons[batch_inds,0,1] = torch.squeeze(wkurt)

        # Autocorrs of low-pass reconstruction 
        auto_corr = texture_utils.weighted_auto_corr_2d(lowpass_rec, spatial_weights, \
                                                        patch_bbox=patch_bbox_square, \
                                                        output_pix = autocorr_output_pix[0], \
                                                        subtract_patch_mean = True, \
                                                        enforce_size=True, device=device)       
        lowpass_recon_autocorrs[batch_inds,0,0:n_autocorr_vals[0]] = \
            torch.reshape(texture_utils.unique_autocorrs(auto_corr), [batch_size_actual, n_autocorr_vals[0]])

        # Looping over spatial frequency/scale
        # Loop goes low SF (smallest npix) to higher SF (largest npix)
        for ff in range(n_sf):
         
            # Scale specific things - get the prf at this resolution of interest    
            n_pix = npix_each_scale[ff+1]           
            spatial_weights = prf_utils.gauss_2d(center=[x,y], sd=sigma, \
                                   patch_size=n_pix, aperture=aperture, dtype=np.float32)
            patch_bbox_square = texture_utils.get_bbox_from_prf(prf_params, spatial_weights.shape,\
                                                                n_prf_sd_out, force_square=True, \
                                                                min_pix=autocorr_output_pix[1+ff])

            # Get the low-pass reconstruction at this scale
            lowpass_summed = fmaps_lowpass_recon[ff+1]  # this is summed over this scale band and those below it
            m, v, wskew, wkurt = texture_utils.get_weighted_pixel_features(lowpass_summed, \
                                                                           spatial_weights, device=device)
            marginal_stats_lowpass_recons[batch_inds,ff+1,0] = torch.squeeze(wskew)
            marginal_stats_lowpass_recons[batch_inds,ff+1,1] = torch.squeeze(wkurt)

            # Autocorrelations of low-pass reconstruction (at this scale)
            auto_corr = texture_utils.weighted_auto_corr_2d(lowpass_summed, spatial_weights, \
                                                            patch_bbox=patch_bbox_square, \
                                                            output_pix = autocorr_output_pix[ff+1], \
                                                            subtract_patch_mean = True, \
                                                            enforce_size=True, device=device)       
            lowpass_recon_autocorrs[batch_inds,ff+1,0:n_autocorr_vals[1+ff]] = \
                torch.reshape(texture_utils.unique_autocorrs(auto_corr), \
                              [batch_size_actual, n_autocorr_vals[1+ff]])

            # Loop over orientation channels
            xx=-1
            for oo1 in range(n_ori):       

                # Magnitude of the complex coefficients; complex cell-like responses
                mag1 = torch.abs(fmaps_complex[ff][:,oo1,:,:].view([batch_size_actual,1,n_pix,n_pix])).float()

                # The mean magnitudes here are basically second-order spectral statistics, 
                # within the specified spatial region defined by weights
                wmean, v, s, k = texture_utils.get_weighted_pixel_features(mag1, \
                                                                           spatial_weights/np.sum(spatial_weights), \
                                                                           device=device)
                mean_magnitudes[batch_inds, ff, oo1] = torch.squeeze(wmean)
                
                mag1 = mag1 - torch.tile(torch.mean(torch.mean(mag1, axis=3, keepdim=True), \
                                                    axis=2, keepdim=True), [1,1,n_pix, n_pix])

                # Real parts of the complex coefficients; simple cell-like responses
                real1 = torch.real(fmaps_complex[ff][:,oo1,:,:].view([batch_size_actual,1,n_pix,n_pix])).float()    
                
                # Average of the real parts within the specified spatial region
                wmean, v, s, k = texture_utils.get_weighted_pixel_features(real1, \
                                                                           spatial_weights/np.sum(spatial_weights), \
                                                                           device=device)
                mean_realparts[batch_inds, ff, oo1] = torch.squeeze(wmean)

                # Complex cell autocorrelation (correlation w spatially shifted versions of itself)     
                auto_corr = texture_utils.weighted_auto_corr_2d(mag1, spatial_weights, \
                                                                patch_bbox=patch_bbox_square, \
                                                                output_pix = autocorr_output_pix[ff+1], \
                                                                subtract_patch_mean = True, \
                                                                enforce_size=True, device=device)       
                magnitude_feature_autocorrs[batch_inds,ff,oo1,0:n_autocorr_vals[1+ff]] = \
                    torch.reshape(texture_utils.unique_autocorrs(auto_corr), [batch_size_actual, n_autocorr_vals[1+ff]])

                # Within-scale correlations - comparing resp at orient==oo1 to responses at all other orientations, same scale.
                for oo2 in np.arange(oo1+1, n_ori):            
                    xx = xx+1 
                    assert(oo1==ori_pairs[xx,0] and oo2==ori_pairs[xx,1])

                    # Magnitude at the other orientation (oo2)
                    mag2 = torch.abs(fmaps_complex[ff][:,oo2,:,:].view([batch_size_actual,1,n_pix,n_pix])).float()      
                    mag2 = mag2 - torch.tile(torch.mean(torch.mean(mag2, axis=3, keepdim=True), \
                                                        axis=2, keepdim=True), [1,1,n_pix, n_pix])

                    # Correlate the magnitude feature maps for the two orientations, within scale
                    cross_corr = texture_utils.weighted_cross_corr_2d(mag1, mag2, spatial_weights, \
                                                                      patch_bbox=None, subtract_patch_mean = True, \
                                                                      device=device)/(n_pix*n_pix)
                    magnitude_within_scale_crosscorrs[batch_inds,ff,xx] = torch.squeeze(cross_corr);

                    # Real part at the other orientation (oo2)
                    real2 = torch.real(fmaps_complex[ff][:,oo2,:,:].view([batch_size_actual,1,n_pix,n_pix])).float()                     

                    # Correlate the real feature maps for the two orientations, within scale
                    cross_corr = texture_utils.weighted_cross_corr_2d(real1, real2, spatial_weights, \
                                                                      patch_bbox=None, subtract_patch_mean = True, \
                                                                      device=device)/(n_pix*n_pix)
                    real_within_scale_crosscorrs[batch_inds,ff,xx] = torch.squeeze(cross_corr);

                # Cross-scale correlations - for these we care about same ori to same ori, so looping over all orientations.
                # Going to compare coefficients at the current scale to those at a coarser scale (ff-1)
                # If we're at first scale (ff=0), then will use a different method.
                if ff>0:

                    for oo2 in range(n_ori):

                        # Get magnitude of coefficients for neighboring (coarser) scale                        
                        mag_coarser2 = \
                            torch.abs(fmaps_coarser_upsampled[ff][:,oo2,:,:].view([batch_size_actual,1,n_pix,n_pix])).float()
                        mag_coarser2 = mag_coarser2 - torch.tile(torch.mean(torch.mean(mag_coarser2, axis=3, \
                                                                           keepdim=True), axis=2, keepdim=True), \
                                                                           [1,1,n_pix, n_pix])

                        # Correlate this with the finer scale
                        cross_corr = texture_utils.weighted_cross_corr_2d(mag1, mag_coarser2, spatial_weights, \
                                                                          patch_bbox=None, subtract_patch_mean = True, \
                                                                          device=device)/(n_pix*n_pix)            
                        magnitude_across_scale_crosscorrs[batch_inds,ff-1,oo1,oo2] = torch.squeeze(cross_corr)

                        # Get the real and imaginary parts at coarser scale
                        real_coarser2 = \
                            torch.real(fmaps_coarser_upsampled[ff][:,oo2,:,:].view([batch_size_actual,1,n_pix,n_pix])).float()
                        imag_coarser2 = \
                            torch.imag(fmaps_coarser_upsampled[ff][:,oo2,:,:].view([batch_size_actual,1,n_pix,n_pix])).float()

                        # Correlate each of these with real part at finer scale
                        cross_corr = texture_utils.weighted_cross_corr_2d(real1, real_coarser2, spatial_weights, \
                                                                          patch_bbox=None, subtract_patch_mean = True, \
                                                                          device=device)/(n_pix*n_pix) 
                        real_imag_across_scale_crosscorrs[batch_inds,ff-1,0,oo1,oo2] = torch.squeeze(cross_corr)

                        cross_corr = texture_utils.weighted_cross_corr_2d(real1, imag_coarser2, spatial_weights, \
                                                                          patch_bbox=None, subtract_patch_mean = True, \
                                                                          device=device)/(n_pix*n_pix) 
                        real_imag_across_scale_crosscorrs[batch_inds,ff-1,1,oo1,oo2] = torch.squeeze(cross_corr)

                else:

                    # instead of different orientations for the "parent" level here, have spatially shifted versions.
                    # this is the up-sampled version of residual lowpass coefficients
                    real_coarser = \
                        torch.real(fmaps_coarser_upsampled[ff][:,0,:,:].view([batch_size_actual,1,n_pix,n_pix])).float()

                    shifts = [[0,0],[1,3],[-1,3],[1,2],[-1,2]]
                    for si1, shift1 in enumerate(shifts):

                        ss,dd = shift1
                        real_coarser_shifted1 = torch.roll(real_coarser, shifts=ss, dims=dd)               
                        # Real part at the finer scale compared to spatially shifted at the coarser scale
                        cross_corr = texture_utils.weighted_cross_corr_2d(real1, real_coarser_shifted1, spatial_weights, \
                                                                          patch_bbox=None, subtract_patch_mean = True, \
                                                                          device=device)/(n_pix*n_pix) 
                        real_spatshift_across_scale_crosscorrs[batch_inds,ff,oo1,si1] = torch.squeeze(cross_corr)

                        for si2 in np.arange(si1+1, n_spatshifts):

                            ss,dd = shifts[si2]
                            real_coarser_shifted2 = torch.roll(real_coarser, shifts=ss, dims=dd) 
                            # Real parts at same scale, comparing spatially shifted.
                            cross_corr = texture_utils.weighted_cross_corr_2d(real_coarser_shifted1, \
                                                                              real_coarser_shifted2, \
                                                                              spatial_weights, patch_bbox=None, \
                                                                              subtract_patch_mean = True, \
                                                                              device=device)/(n_pix*n_pix) 
                            real_spatshift_within_scale_crosscorrs[batch_inds,ff,si1,si2] = torch.squeeze(cross_corr)

            
    if not keep_orig_shape:
        # Reshape everything to [ntrials x nfeatures]

        mean_magnitudes = torch.reshape(mean_magnitudes, [n_trials, -1])
        mean_realparts = torch.reshape(mean_realparts, [n_trials, -1])
        marginal_stats_lowpass_recons = torch.reshape(marginal_stats_lowpass_recons, [n_trials, -1])
        variance_highpass_resid =torch.reshape(variance_highpass_resid, [n_trials, -1])

        magnitude_feature_autocorrs = torch.reshape(magnitude_feature_autocorrs, [n_trials, -1])
        # take out the zero columns, which happen because of different size autocorr outputs.
        magnitude_feature_autocorrs = magnitude_feature_autocorrs[:,torch.sum(magnitude_feature_autocorrs, axis=0)!=0]
        assert(magnitude_feature_autocorrs.shape[1]==np.sum(n_autocorr_vals[1:]*n_ori))

        lowpass_recon_autocorrs = torch.reshape(lowpass_recon_autocorrs, [n_trials, -1])
        lowpass_recon_autocorrs = lowpass_recon_autocorrs[:,torch.sum(lowpass_recon_autocorrs, axis=0)!=0]
        assert(lowpass_recon_autocorrs.shape[1]==np.sum(n_autocorr_vals))

        highpass_resid_autocorrs = torch.reshape(highpass_resid_autocorrs, [n_trials, -1])

        magnitude_within_scale_crosscorrs = torch.reshape(magnitude_within_scale_crosscorrs, [n_trials, -1])
        real_within_scale_crosscorrs = torch.reshape(real_within_scale_crosscorrs, [n_trials, -1])
        magnitude_across_scale_crosscorrs = torch.reshape(magnitude_across_scale_crosscorrs, [n_trials, -1])
        real_imag_across_scale_crosscorrs = torch.reshape(real_imag_across_scale_crosscorrs, [n_trials, -1])
        real_spatshift_within_scale_crosscorrs = torch.reshape(real_spatshift_within_scale_crosscorrs, [n_trials, -1])

        real_spatshift_within_scale_crosscorrs = \
            real_spatshift_within_scale_crosscorrs[:,torch.sum(real_spatshift_within_scale_crosscorrs, axis=0)!=0]
        assert(real_spatshift_within_scale_crosscorrs.shape[1]==np.sum(np.arange(1,n_spatshifts)))

        real_spatshift_across_scale_crosscorrs = torch.reshape(real_spatshift_across_scale_crosscorrs, [n_trials, -1])

        
    return  pixel_stats, \
            mean_magnitudes, \
            mean_realparts, \
            marginal_stats_lowpass_recons, \
            variance_highpass_resid, \
            magnitude_feature_autocorrs, \
            lowpass_recon_autocorrs, \
            highpass_resid_autocorrs, \
            magnitude_within_scale_crosscorrs, \
            real_within_scale_crosscorrs, \
            magnitude_across_scale_crosscorrs, \
            real_imag_across_scale_crosscorrs, \
            real_spatshift_within_scale_crosscorrs, \
            real_spatshift_across_scale_crosscorrs

