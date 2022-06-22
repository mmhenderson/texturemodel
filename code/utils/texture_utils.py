import numpy as np
import torch
from utils import torch_utils

def get_weighted_pixel_features(image_batch, spatial_weights, device=None):
    """
    Compute mean, variance, skewness, kurtosis of luminance values for each of a batch of images.
    Input size is [batch_size x n_channels x npix x npix]
    Spatial weights describes a weighting function, [npix x npix]
    Returns [batch_size x n_channels] size array for each property.
    """
    
    if isinstance(image_batch, np.ndarray):
        image_batch = torch_utils._to_torch(image_batch, device).contiguous()
    if isinstance(spatial_weights, np.ndarray):
        spatial_weights = torch_utils._to_torch(spatial_weights, device)
     
    assert(image_batch.shape[2]==spatial_weights.shape[0] and image_batch.shape[3]==spatial_weights.shape[1])
    assert(image_batch.shape[1]==1)
    
    batch_size = image_batch.shape[0]
    n_channels = image_batch.shape[1]
    n_pix = image_batch.shape[2]

    image_batch = image_batch.view([batch_size, n_channels, n_pix**2])
    spatial_weights = spatial_weights/torch.sum(spatial_weights)
    spatial_weights = spatial_weights.view([1,1,n_pix**2]).expand([batch_size,n_channels,-1]) # [batch_size x n_channels x nPix x nPix]    
   
    ims_weighted = image_batch * spatial_weights
   
    wmean = torch.sum(ims_weighted, axis=2).view([batch_size,-1,1])

    wvar = torch.sum(spatial_weights * (image_batch - wmean.expand([-1,-1,n_pix**2]))**2, axis=2).view([batch_size,-1,1])
    wvar += 10**(-6) # putting this in to prevent runaway values - can get massive skew/kurt if this is tiny.

    wskew = torch.sum(spatial_weights *(image_batch - wmean.expand([-1,-1,n_pix**2]))**3 / (wvar**(3/2)), axis=2).view([batch_size,-1,1])
    
    wkurt = torch.sum(spatial_weights *(image_batch - wmean.expand([-1,-1,n_pix**2]))**4 / (wvar**(2)), axis=2).view([batch_size,-1,1])
   
    # correct for nans/inf values which happen when variance is very small (denominator)
    wskew[torch.isnan(wskew)] = 0.0
    wkurt[torch.isnan(wkurt)] = 0.0
    wskew[torch.isinf(wskew)] = 0.0
    wkurt[torch.isinf(wkurt)] = 0.0
    
    return torch.squeeze(wmean, dim=2), torch.squeeze(wvar, dim=2), torch.squeeze(wskew, dim=2), torch.squeeze(wkurt, dim=2)


def weighted_auto_corr_2d(images, spatial_weights, patch_bbox=None, output_pix=None, subtract_patch_mean=False, enforce_size=False, device=None):

    """
    Compute autocorrelation of a batch of images, weighting the pixels based on the values in spatial_weights (could be for instance a pRF definition for a voxel).
    Can optionally specify a square patch of the image to compute over, based on "patch_bbox" params. Otherwise use whole image.
    Using fft method to compute, should be fast.
    Input parameters:
        patch_bbox: (optional) bounding box of the patch to use for this calculation. [xmin xmax ymin ymax], see get_bbox_from_prf
        output_pix: the size of the autocorrelation matrix output by this function. If this is an even number, the output size is this value +1. Achieved by cropping out the center of the final autocorrelation 
            matrix  (note that the full image patch is still used in computing the autocorrelation, but just the center values are returned).
            If None, then returns the full autocorrelation matrix (same size as image patch.)
        subtract_patch_mean: subtract weighted mean of image before computing autocorr?
        enforce_size: if image patch is smaller than desired output, should we pad w zeros so that it has to be same size?
    Returns:
        A matrix describing the correlation of the image and various spatially shifted versions of it.
    """
    
    
    if device is None:
        device = torch.device('cpu:0')        
    if isinstance(images, np.ndarray):
        images = torch_utils._to_torch(images, device)
    if isinstance(spatial_weights, np.ndarray):
        spatial_weights = torch_utils._to_torch(spatial_weights, device)
            
    if len(np.shape(images))==2:
        # pretend the batch and channel dims exist, for 2D input only (3D won't work)
        single_image=True
        images = images.view([1,1,images.shape[0],-1])
    else:
        single_image=False
        
    # have to be same size
    assert(images.shape[2]==spatial_weights.shape[0] and images.shape[3]==spatial_weights.shape[1])
    # images is [batch_size x n_channels x nPix x nPix]
    batch_size = images.shape[0]
    n_channels = images.shape[1]    
   
    sum_weights_full = torch.sum(spatial_weights)

    if patch_bbox is not None:    
        [xmin, xmax, ymin, ymax] = patch_bbox
        # first crop out the region of the image that's currently of interest
        images = images[:,:,xmin:xmax, ymin:ymax]
        # crop same region from spatial weights matrix
        spatial_weights = spatial_weights[xmin:xmax, ymin:ymax]

    sum_weights = torch.sum(spatial_weights)

    spatial_weights = spatial_weights.view([1,1,spatial_weights.shape[0],-1]).expand([batch_size,n_channels,-1,-1]) # [batch_size x n_channels x nPix x nPix]    
    
    # compute autocorrelation of this image patch
    if subtract_patch_mean:

        wmean = torch.sum(torch.sum(images * spatial_weights/sum_weights, dim=3), dim=2) # size is [batch_size x 1]
        wmean = wmean.view([batch_size,-1,1,1]).expand([-1,-1,images.shape[2],images.shape[3]]) # [batch_size x n_channels x nPix x nPix]
        
        weighted_images = (images - wmean) * torch.sqrt(spatial_weights/sum_weights_full) # square root of the weights here because they will get squared again in next operation

        auto_corr = torch.fft.fftshift(torch.real(torch.fft.ifft2(torch.abs(torch.fft.fft2(weighted_images, dim=[2,3]))**2, dim=[2,3])), dim=[2,3]);
    else:
        weighted_images = images * torch.sqrt(spatial_weights_full/sum_weights_full)
        auto_corr = torch.fft.fftshift(torch.real(torch.fft.ifft2(torch.abs(torch.fft.fft2(weighted_images, dim=[2,3]))**2, dim=[2,3])), dim=[2,3]);

    if output_pix is not None:

        # crop out just the center region
        new_center = int(np.floor(auto_corr.shape[2]/2))
        n_pix_out = np.min([int(np.floor(output_pix/2)), np.min([new_center, auto_corr.shape[2]-new_center])])
        auto_corr = auto_corr[:,:,new_center-n_pix_out:new_center+n_pix_out+1, new_center-n_pix_out:new_center+n_pix_out+1]        
    
    if enforce_size and not (np.shape(auto_corr)[2]==output_pix or np.shape(auto_corr)[2]==output_pix+1):
        
        # just pad w zeros if want same size.
        pix_diff = output_pix - np.shape(auto_corr)[2]   
        auto_corr = torch.nn.functional.pad(auto_corr, [int(np.floor(pix_diff/2)), int(np.ceil(pix_diff/2)), int(np.floor(pix_diff/2)), int(np.ceil(pix_diff/2))], mode='constant', value=0)
        assert(np.shape(auto_corr)[2]==output_pix and np.shape(auto_corr)[3]==output_pix)

    if single_image:
        auto_corr = torch.squeeze(auto_corr)
        
    return auto_corr


def weighted_cross_corr_2d(images1, images2, spatial_weights, patch_bbox=None, subtract_patch_mean=True, device=None):

    """
    Compute cross-correlation of two identically-sized images, weighting the pixels based on the values in spatial_weights (could be for instance a pRF definition for a voxel).
    Can optionally specify a square patch of the image to compute over, based on "patch_bbox" params. Otherwise use whole image.
    Basically a dot product of image values.
    Input parameters:
        patch_bbox: (optional) bounding box of the patch to use for this calculation. [xmin xmax ymin ymax], see get_bbox_from_prf
        subtract_patch_mean: do you want to subtract the weighted mean of image patch before computing?
    Returns:
        A single value that captures correlation between images (zero spatial shift)
            
    """
    
    if device is None:
        device = torch.device('cpu:0')  
    if isinstance(images1, np.ndarray):
        images1 = torch_utils._to_torch(images1, device)
    if isinstance(images2, np.ndarray):
        images2 = torch_utils._to_torch(images2, device)
    if isinstance(spatial_weights, np.ndarray):
        spatial_weights = torch_utils._to_torch(spatial_weights, device)      
    
    if len(np.shape(images1))==2:
        # pretend the batch and channel dims exist, for 2D input only (3D won't work)
        single_image=True
        images1 = images1.view([1,1,images1.shape[0],-1])
        images2 = images2.view([1,1,images2.shape[0],-1])
    else:
        single_image=False
        
    # have to be same size
    assert(images1.shape==images2.shape)
    assert(images1.shape[2]==spatial_weights.shape[0] and images1.shape[3]==spatial_weights.shape[1])
    assert(images2.shape[2]==spatial_weights.shape[0] and images2.shape[3]==spatial_weights.shape[1])
    # images is [batch_size x n_channels x nPix x nPix]
    batch_size = images1.shape[0]
    n_channels = images1.shape[1]
    
    sum_weights_full = torch.sum(spatial_weights)

    if patch_bbox is not None:

        [xmin, xmax, ymin, ymax] = patch_bbox
        # first crop out the region of the image that's currently of interest
        images1 = images1[:,:,xmin:xmax, ymin:ymax]
        images2 = images2[:,:,xmin:xmax, ymin:ymax]
        # crop same region from spatial weights matrix
        spatial_weights = spatial_weights[xmin:xmax, ymin:ymax]

    sum_weights = torch.sum(spatial_weights)

    spatial_weights = spatial_weights.view([1,1,spatial_weights.shape[0],-1]).expand([batch_size,n_channels,-1,-1]) # [batch_size x n_channels x nPix x nPix]    

    # compute cross-correlation
    if subtract_patch_mean:
        # subtract mean of each weighted image patch and take their dot product.
        # this quantity is equal to weighted covariance (only true if mean-centered)
        wmean1 = torch.sum(torch.sum(images1 * spatial_weights/sum_weights, dim=3), dim=2) # size is [batch_size x 1]
        wmean1 = wmean1.view([batch_size,-1,1,1]).expand([-1,-1,images1.shape[2],images1.shape[3]]) # [batch_size x n_channels x nPix x nPix]
        wmean2 = torch.sum(torch.sum(images2 * spatial_weights/sum_weights, dim=3), dim=2) # size is [batch_size x 1]
        wmean2 = wmean2.view([batch_size,-1,1,1]).expand([-1,-1,images2.shape[2],images2.shape[3]]) # [batch_size x n_channels x nPix x nPix]
        weighted_images1 = (images1 - wmean1) * torch.sqrt(spatial_weights/sum_weights_full) # square root of the weights here because they will get squared again in dot product operation.
        weighted_images2 = (images2 - wmean2) * torch.sqrt(spatial_weights/sum_weights_full)

        cross_corr = torch.sum(torch.sum(weighted_images1 * weighted_images2, dim=3), dim=2)    

    else:
        # dot product of raw (weighted) values
        # this is closer to what scipy.signal.correlate2d will do (except this is weighted)
        weighted_images1 = images1 * torch.sqrt(spatial_weights/sum_weights_full)
        weighted_images2 = images2 * torch.sqrt(spatial_weights/sum_weights_full)
        cross_corr = torch.sum(torch.sum(weighted_images1 * weighted_images2, dim=3), dim=2)      
        
    if single_image:
        cross_corr = torch.squeeze(cross_corr)
        
    return cross_corr


def get_bbox_from_prf(prf_params, image_size, n_prf_sd_out=2, min_pix=None, verbose=False, force_square=False):
    """
    For a given pRF center and size, calculate the square bounding box that captures a specified number of SDs from the center (default=2 SD)
    Returns [xmin, xmax, ymin, ymax]
    Input image has to be square.
    """
    x,y,sigma = prf_params
    n_pix = image_size[0]
    assert(image_size[1]==n_pix)
    assert(sigma>0 and n_prf_sd_out>0)
    if min_pix is not None:
        assert(min_pix<=n_pix)
    else:
        min_pix=1
        
    # decide on the window to use for correlations, based on prf parameters. Patch goes # SD from the center (2 by default).
    # note this can't be < 1, even for the smallest choice of parameters (since rounding up). this way it won't be too small.
    pix_from_center = int(np.ceil(sigma*n_prf_sd_out*n_pix))   
    pix_from_center = int(np.max([pix_from_center, np.floor(min_pix/2)]))

    # center goes [row ind, col ind]
    center = np.array((n_pix/2  - y*n_pix, x*n_pix + n_pix/2)) # note that the x/y dims get swapped here because of how pRF parameters are defined.
    if np.ceil(center[0])==np.floor(center[0]):
        center[0] = center[0]+0.0001
    if np.ceil(center[1])==np.floor(center[1]):
        center[1] = center[1]+0.0001

    # now defining the extent of the bbox. want to err on the side of making it too big, so taking floor/ceiling...
    xmin = int(np.floor(center[0]-pix_from_center))
    xmax = int(np.ceil(center[0]+pix_from_center))
    ymin = int(np.floor(center[1]-pix_from_center))
    ymax = int(np.ceil(center[1]+pix_from_center))

    # cropping it to within the image bounds. Can end up being a rectangle rather than square.
    [xmin, xmax, ymin, ymax] = np.maximum(np.minimum([xmin, xmax, ymin, ymax], n_pix), 0)

    # Did the crop make it smaller than it should be?
    if xmax-xmin<min_pix:
        if xmin==0:
            xmax = min_pix
        else:
            xmin = n_pix - min_pix
    if ymax-ymin<min_pix:
        if ymin==0:
            ymax = min_pix
        else:
            ymin = n_pix - min_pix
      
    # decide if we want square or are ok with a rectangle
    minside = np.min([xmax-xmin, ymax-ymin])
    maxside = np.max([xmax-xmin, ymax-ymin])
    if minside!=maxside and force_square:

      
        if min_pix is None or minside>=min_pix:
            # will trim the box to make it smaller but square.
            if verbose:
                print('trimming bbox to make it square')
                print('original bbox was:')
                print([xmin, xmax, ymin, ymax])

            n2trim = [int(np.floor((maxside-minside)/2)), int(np.ceil((maxside-minside)/2))]
            if np.argmin([xmax-xmin, ymax-ymin])==0:
                ymin = ymin+n2trim[0]
                ymax = ymax-n2trim[1]
            else:
                xmin = xmin+n2trim[0]
                xmax = xmax-n2trim[1]                
        else:
            # instead of trimming, we'll move the box over until it all fits.
            if verbose:
                print('moving bbox to make it square')
                print('original bbox was:')
                print([xmin, xmax, ymin, ymax])

            if np.argmin([xmax-xmin, ymax-ymin])==0:
                if xmin==0:
                    xmax = xmax+maxside-minside
                else:
                    xmin = xmin-(maxside-minside)
            else:
                if ymin==0:
                    ymax = ymax+maxside-minside                       
                else:
                    ymin = ymin-(maxside-minside)

        assert((xmax-xmin)==(ymax-ymin))
  
    if verbose:
        print('final bbox will be:')
        print([xmin, xmax, ymin, ymax])
        
    # checking to see if the patch has become just one pixel. this can happen due to the cropping.
    # if this happens, cross-correlations will give zero.
    if ((xmax-xmin)<2 or (ymax-ymin)<2):
        print('Warning: your patch only has one pixel (for n_pix: %d and prf params: [%.2f, %.2f, %.2f])\n'%(n_pix,x,y,sigma))      
        
    return [xmin, xmax, ymin, ymax]


def unique_autocorrs(acor):

    """
    Pick out the unique values from (symmetric) 2D autocorrelation matrix.
    This works on batches of size [batch_size x nchannels x height x width]
    Select unique values from last two dims.
    Works w tensor input for acor.
    """
    height = acor.shape[2]
    inds1, inds2 = np.triu_indices(height,1)
    unvals = acor[:,:,inds1, inds2]
    unvals2 = acor[:,:,np.arange(0,int(np.ceil(height/2))), np.arange(0,int(np.ceil(height/2)))]
    unvals = torch.cat([unvals, unvals2], axis=2)
    assert(unvals.shape[2]==int((height**2+1)/2))
#     assert(torch.numel(torch.unique(unvals))==torch.numel(torch.unique(acor)))

    return unvals

def double_phase(fmap):
    
    """
    Double the phase (i.e. angle) values of a complex array
    modified by MH from https://github.com/LabForComputationalVision/textureSynth
    """     
    
    rtmp = np.real(fmap)
    itmp = np.imag(fmap)

    theta = np.arctan2(itmp, rtmp) # first get original phase (angle)
    rad = np.sqrt(rtmp**2 + itmp**2) # then get original magnitude

    # then put back together, using a*e^(ib)
    fmap_phase_doubled = rad * np.exp(2 * 1j*theta) 

    return fmap_phase_doubled

def expand(image_batch, factor):

    """
    Expand spatially an image in a factor f in X and in Y.
    Image may be complex.
    It fills in with zeros in the Fourier domain.
    image_batch_expanded = expand(image_batch, factor)
    
    This version works for batches of images, OR single.
    Batches should be [batch_size x nchannels x height x width]
    If single image, should be [height x width]
    
    Converted by MH from matlab function, from the library at:
    https://github.com/LabForComputationalVision/textureSynth
    
    % See also: shrink.m
    % JPM, May 95, Instituto de Optica, CSIC, Madrid.

    """

    if round(factor,1)==1.0:
        return image_batch

    if len(image_batch.shape)==2:
        image_batch = np.expand_dims(np.expand_dims(image_batch, axis=0), axis=0)
        single_image=True
    else:
        single_image=False
        
    batch_size, nchans, orig_height, orig_width =image_batch.shape
    new_height = int(orig_height*factor)
    new_width = int(orig_width*factor)
    
    freq_rep_exp = np.full((batch_size, nchans, new_height, new_width),0+0*1j)

    freq_rep = factor**2 * np.fft.fftshift(np.fft.fft2(image_batch, axes=(2,3)), axes=(2,3))
  
    y1 = int(new_height/2 + 2 - new_height/(2*factor))
    y2 = int(new_height/2 + new_height/(2*factor))

    x1 = int(new_width/2 + 2 - new_width/(2*factor))
    x2 = int(new_width/2 + new_width/(2*factor))

    freq_rep_exp[:,:,y1-1:y2, x1-1:x2] = freq_rep[:,:,1:int(new_height/factor), 1:int(new_width/factor)]
    freq_rep_exp[:,:,y1-2, x1-1:x2] = freq_rep[:,:,0, 1:int(new_width/factor)]/2
    freq_rep_exp[:,:,y2, x1-1:x2] = np.conjugate(freq_rep[:,:,0, int(new_width/factor):0:-1]/2)
    freq_rep_exp[:,:,y1-1:y2, x1-2] = freq_rep[:,:,1: int(new_height/factor), 0]/2
    freq_rep_exp[:,:,y1-1:y2, x2] = np.conjugate(freq_rep[:,:,int(new_height/factor):0:-1, 0]/2)

    esq=freq_rep[:,:,0,0]/4;
    freq_rep_exp[:,:,y1-2, x1-2] = esq
    freq_rep_exp[:,:,y1-2, x2] = esq
    freq_rep_exp[:,:,y2, x1-2] = esq
    freq_rep_exp[:,:,y2, x2] = esq

    image_batch_expanded = np.fft.ifft2(np.fft.ifftshift(freq_rep_exp, axes=(2,3)), axes=(2,3));
 
    if np.all(np.imag(image_batch)==0):
        image_batch_expanded = np.real(image_batch_expanded);
                    
    if single_image:
        image_batch_expanded = np.squeeze(np.squeeze(image_batch_expanded, axis=0), axis=0)
    
    return image_batch_expanded


def shrink(image_batch, factor):
    
    """
    Shrink spatially an image in a factor f in X and in Y.
    Image may be complex.
    image_batch_shrunk = shrink(image_batch, factor)
    
    This version works for batches of images, OR single.
    Batches should be [batch_size x nchannels x height x width]
    If single image, should be [height x width]
    
    Converted by MH from matlab function, from the library at:
    https://github.com/LabForComputationalVision/textureSynth
    
    % See also: expand.m
    % JPM, May 95, Instituto de Optica, CSIC, Madrid.

    """

    if round(factor,1)==1.0:
        return image_batch
  
    if len(image_batch.shape)==2:
        image_batch = np.expand_dims(np.expand_dims(image_batch, axis=0), axis=0)
        single_image=True
    else:
        single_image=False
        
    batch_size, nchans, orig_height, orig_width = image_batch.shape
    new_height = int(orig_height/factor)
    new_width = int(orig_width/factor)
    
    freq_rep_shr = np.full((batch_size, nchans, new_height, new_width),0+0*1j)

    freq_rep = 1/factor**2 * np.fft.fftshift(np.fft.fft2(image_batch, axes=(2,3)), axes=(2,3))
   
    y1 = int(orig_height/2 + 2 - orig_height/(2*factor))
    y2 = int(orig_height/2 + orig_height/(2*factor))

    x1 = int(orig_width/2 + 2 - orig_width/(2*factor))
    x2 = int(orig_width/2 + orig_width/(2*factor))

    freq_rep_shr[:,:,1:int(orig_height/factor), 1:int(orig_width/factor)] = freq_rep[:,:,y1-1:y2 ,x1-1:x2]
    freq_rep_shr[:,:,0,1:int(orig_width/factor)]=(freq_rep[:,:,y1-2, x1-1:x2]+freq_rep[:,:,y2, x1-1:x2])/2
    freq_rep_shr[:,:,1:int(orig_height/factor),0] = (freq_rep[:,:,y1-1:y2, x1-2] + freq_rep[:,:,y1-1:y2, x2])/2
    freq_rep_shr[:,:,0,0] = (freq_rep[:,:,y1-2,x1-1] + freq_rep[:,:,y1-2,x2] + freq_rep[:,:,y2, x1-1] + freq_rep[:,:,y2, x2+1])/4

    image_batch_shrunk = np.fft.ifft2(np.fft.ifftshift(freq_rep_shr, axes=(2,3)), axes=(2,3));
    
    if np.all(np.imag(image_batch)==0):
        image_batch_shrunk = np.real(image_batch_shrunk)
    
    if single_image:
        image_batch_shrunk = np.squeeze(np.squeeze(image_batch_shrunk, axis=0), axis=0)
    
    return image_batch_shrunk