import struct
import numpy as np
import math
import copy

def make_log_polar_grid(sigma_range=[0.02, 1], n_sigma_steps=10, \
                          eccen_range=[0, 7/8.4], n_eccen_steps=10, n_angle_steps=16):
    
    """
    Create a grid of pRF positions/sizes that are evenly spaced in polar angle.
    Sizes and eccentricities are logarithmically spaced.
    Units for size/sigma are relative to image aperture size (1.0 units)
    To convert to degrees, multiply these values by degrees of display (8.4 degrees for NSD expts)
    """

    sigma_vals = np.logspace(np.log(sigma_range[0]), np.log(sigma_range[1]), \
                             base=np.e, num=n_sigma_steps)
    # min eccen should usually be zero, accounting for this here
    small_value = 0.10
    eccen_vals = np.logspace(np.log(eccen_range[0]+small_value), np.log(eccen_range[1]+small_value), \
                             base=np.e, num=n_eccen_steps) - small_value
    angle_step = 2*np.pi/n_angle_steps
    angle_vals = np.linspace(0,np.pi*2-angle_step, n_angle_steps)

    eccen_vals = eccen_vals.astype(np.float32)
    angle_vals = angle_vals.astype(np.float32)
    sigma_vals = sigma_vals.astype(np.float32)

    # First, create the grid of all possible combinations.
    x_vals = (eccen_vals[:,np.newaxis] * np.cos(angle_vals[np.newaxis,:]))
    y_vals = (eccen_vals[:,np.newaxis] * np.sin(angle_vals[np.newaxis,:]))

    x_vals = np.tile(np.reshape(x_vals, [n_angle_steps*n_eccen_steps, 1]), [n_sigma_steps,1])
    y_vals = np.tile(np.reshape(y_vals, [n_angle_steps*n_eccen_steps, 1]), [n_sigma_steps,1])

    sigma_vals = np.repeat(sigma_vals, n_angle_steps*n_eccen_steps)[:,np.newaxis]

    prf_params = np.concatenate([x_vals, y_vals, sigma_vals], axis=1)

    # Now removing a few of these that we don't want to actually use - some duplicates, 
    # and some that go entirely outside the image region.
    unrows, inds = np.unique(prf_params, axis=0, return_index=True)
    prf_params = prf_params[np.sort(inds),:]

    # what is the approx spatial extent of the pRF? Assume 1 standard deviation.
    n_std = 1
    left_extent = prf_params[:,0] - prf_params[:,2]*n_std
    right_extent = prf_params[:,0] + prf_params[:,2]*n_std
    top_extent = prf_params[:,1] + prf_params[:,2]*n_std
    bottom_extent = prf_params[:,1] - prf_params[:,2]*n_std

    out_of_bounds = (left_extent > 0.5) | (right_extent < -0.5) | (top_extent < -0.5) | (bottom_extent > 0.5)

    prf_params = prf_params[~out_of_bounds,:]

    return prf_params

def make_log_polar_grid_scale_size_eccen(eccen_range=[0, 7/8.4], n_eccen_steps = 10, n_angle_steps = 16):

    small_value = 0.10
    eccen_vals = np.logspace(np.log(eccen_range[0]+small_value), np.log(eccen_range[1]+small_value), \
                             base=np.e, num=n_eccen_steps) - small_value
    # Assuming an approximate fixed relationship between eccentricity and the range of sizes that can occur.
    # there will be three possible sizes per eccen, to account for the range of size/eccen slopes
    # across ROIs.
    # these params are approximated based on actual pRF fits from several papers.
    n_sigma_per_eccen = 3;
    sc = 0.5
    est_sizes = (eccen_vals*0.7)+0.05
    min_sizes = est_sizes - sc*est_sizes
    max_sizes = est_sizes + sc*est_sizes

    sigma_vals = np.concatenate([min_sizes[:,np.newaxis],est_sizes[:,np.newaxis],\
                                 max_sizes[:,np.newaxis]], axis=1)
    sigma_vals = sigma_vals.ravel()
    eccen_vals = np.repeat(eccen_vals,n_sigma_per_eccen)

    n_angle_steps = 16;
    angle_step = 2*np.pi/n_angle_steps
    angle_vals = np.linspace(0,np.pi*2-angle_step, n_angle_steps)

    eccen_vals = eccen_vals.astype(np.float32)
    angle_vals = angle_vals.astype(np.float32)
    sigma_vals = sigma_vals.astype(np.float32)

    # Create the grid of all possible combinations.
    x_vals = (eccen_vals[:,np.newaxis] * np.cos(angle_vals[np.newaxis,:]))
    y_vals = (eccen_vals[:,np.newaxis] * np.sin(angle_vals[np.newaxis,:]))

    x_vals = np.reshape(x_vals, [n_angle_steps*n_eccen_steps*n_sigma_per_eccen, 1])
    y_vals = np.reshape(y_vals, [n_angle_steps*n_eccen_steps*n_sigma_per_eccen, 1])

    sigma_vals = np.repeat(sigma_vals, n_angle_steps)[:,np.newaxis]
    eccen_vals = np.repeat(eccen_vals, n_angle_steps)[:,np.newaxis]

    prf_params = np.array([x_vals,y_vals,sigma_vals])[:,:,0].T

    # what is the approx spatial extent of the pRF? Assume 1 standard deviation.
    n_std = 1
    left_extent = prf_params[:,0] - prf_params[:,2]*n_std
    right_extent = prf_params[:,0] + prf_params[:,2]*n_std
    top_extent = prf_params[:,1] + prf_params[:,2]*n_std
    bottom_extent = prf_params[:,1] - prf_params[:,2]*n_std

    out_of_bounds = (left_extent > 0.5) | (right_extent < -0.5) | (top_extent < -0.5) | (bottom_extent > 0.5)
    prf_params = prf_params[~out_of_bounds,:]
    
    return prf_params


def make_polar_angle_grid(sigma_range=[0.04, 1], n_sigma_steps=12, \
                          eccen_range=[0, 1.4], n_eccen_steps=12, n_angle_steps=16):
    
    """
    Create a grid of pRF positions/sizes that are evenly spaced in polar angle.
    Sizes and eccentricities are logarithmically spaced.
    Units for size/sigma are relative to image aperture size (1.0 units)
    To convert to degrees, multiply these values by degrees of display (8.4 degrees for NSD expts)
    """

    sigma_vals = np.logspace(np.log10(sigma_range[0]), np.log10(sigma_range[1]), n_sigma_steps)
    # min eccen should usually be zero, accounting for this here
    eccen_vals = np.logspace(np.log10(eccen_range[0]+0.1), np.log10(eccen_range[1]+0.1), n_eccen_steps) - 0.1
    angle_step = 2*np.pi/n_angle_steps
    angle_vals = np.linspace(0,np.pi*2-angle_step, n_angle_steps)

    eccen_vals = eccen_vals.astype(np.float32)
    angle_vals = angle_vals.astype(np.float32)
    sigma_vals = sigma_vals.astype(np.float32)

    # First, create the grid of all possible combinations.
    x_vals = (eccen_vals[:,np.newaxis] * np.cos(angle_vals[np.newaxis,:]))
    y_vals = (eccen_vals[:,np.newaxis] * np.sin(angle_vals[np.newaxis,:]))

    x_vals = np.tile(np.reshape(x_vals, [n_angle_steps*n_eccen_steps, 1]), [n_sigma_steps,1])
    y_vals = np.tile(np.reshape(y_vals, [n_angle_steps*n_eccen_steps, 1]), [n_sigma_steps,1])

    sigma_vals = np.repeat(sigma_vals, n_angle_steps*n_eccen_steps)[:,np.newaxis]

    prf_params = np.concatenate([x_vals, y_vals, sigma_vals], axis=1)

    # Now removing a few of these that we don't want to actually use - some duplicates, 
    # and some that go entirely outside the image region.
    unrows, inds = np.unique(prf_params, axis=0, return_index=True)
    prf_params = prf_params[np.sort(inds),:]

    # what is the approx spatial extent of the pRF? Assume 1 standard deviation.
    n_std = 1
    left_extent = prf_params[:,0] - prf_params[:,2]*n_std
    right_extent = prf_params[:,0] + prf_params[:,2]*n_std
    top_extent = prf_params[:,1] + prf_params[:,2]*n_std
    bottom_extent = prf_params[:,1] - prf_params[:,2]*n_std

    out_of_bounds = (left_extent > 0.5) | (right_extent < -0.5) | (top_extent < -0.5) | (bottom_extent > 0.5)

    prf_params = prf_params[~out_of_bounds,:]

    return prf_params

def make_rect_grid(sigma_range=[0.02, 0.10], n_sigma_steps=10, min_grid_spacing=0.04):
    
    sigma_vals = np.logspace(np.log(sigma_range[0]), np.log(sigma_range[1]), \
                             base=np.e, num=n_sigma_steps)

    extend_xy = np.max(sigma_vals)/2
    n_grid_pts = int(np.ceil(1/min_grid_spacing))
    x_vals = np.linspace(-0.5-extend_xy, 0.5+extend_xy, n_grid_pts)
    y_vals = np.linspace(-0.5-extend_xy, 0.5+extend_xy, n_grid_pts)
    x_vals, y_vals = np.meshgrid(x_vals, y_vals);

    x_vals = np.tile(np.reshape(x_vals, [n_grid_pts**2,1]), [n_sigma_steps,1])
    y_vals = np.tile(np.reshape(y_vals, [n_grid_pts**2,1]), [n_sigma_steps,1])

    sigma_vals = np.repeat(sigma_vals, n_grid_pts**2)[:,np.newaxis]

    prf_params = np.concatenate([x_vals, y_vals, sigma_vals], axis=1)

    # Now removing a few of these that we don't want to actually use - 
    # some that go entirely outside the image region.
    unrows, inds = np.unique(prf_params, axis=0, return_index=True)
    prf_params = prf_params[np.sort(inds),:]

    # what is the approx spatial extent of the pRF? Assume 1 standard deviation.
    n_std = 1
    left_extent = prf_params[:,0] - prf_params[:,2]*n_std
    right_extent = prf_params[:,0] + prf_params[:,2]*n_std
    top_extent = prf_params[:,1] + prf_params[:,2]*n_std
    bottom_extent = prf_params[:,1] - prf_params[:,2]*n_std

    out_of_bounds = (left_extent > 0.5) | (right_extent < -0.5) | (top_extent < -0.5) | (bottom_extent > 0.5)

    prf_params = prf_params[~out_of_bounds,:]

    
    return prf_params


class subdivision_1d(object):
    def __init__(self, n_div=1, dtype=np.float32):
        self.length = n_div
        self.dtype = dtype
        
    def __call__(self, center, width):
        '''	returns a list of point positions '''
        return [center] * self.length
    
class linspace(subdivision_1d):    
    def __init__(self, n_div, right_bound=False, dtype=np.float32, **kwargs):
        super(linspace, self).__init__(n_div, dtype=np.float32, **kwargs)
        self.__rb = right_bound
        
    def __call__(self, center, width):
        if self.length<=1:
            return [center]     
        if self.__rb:
            d = np.float32(width)/(self.length-1)
            vmin, vmax = center, center+width  
        else:
            d = np.float32(width)/self.length
            vmin, vmax = center+(d-width)/2, center+width/2 
        return np.arange(vmin, vmax+1e-12, d).astype(dtype=self.dtype)
    
class logspace(subdivision_1d):    
    def __init__(self, n_div, dtype=np.float32, **kwargs):
        super(logspace, self).__init__(n_div, dtype=np.float32, **kwargs)
               
    def __call__(self, start, stop):    
        if self.length <= 1:
            return [start]
        lstart = np.log(start+1e-12)
        lstop = np.log(stop+1e-12)
        dlog = (lstop-lstart)/(self.length-1)
        return np.exp(np.arange(lstart, lstop+1e-12, dlog)).astype(self.dtype)

def model_space_pyramid(sigmas, min_spacing, aperture):
    rf = []
    for s in sigmas:
        X, Y = np.meshgrid(np.linspace(-aperture/2, aperture/2, int(np.ceil(aperture/(s * min_spacing)))),
                           np.linspace(-aperture/2, aperture/2, int(np.ceil(aperture/(s * min_spacing)))))
        rf += [np.stack([X.flatten(), Y.flatten(), np.full(fill_value=s, shape=X.flatten().shape)], axis=1),]
    return np.concatenate(rf, axis=0)

def model_space_pyramid2(sigmas, min_spacing, aperture):
    rf = []
    n_grid_list = [21,15,11,9,7,7,7,7,7]
    for si, s in enumerate(sigmas):
        n_grid = n_grid_list[si]
        X, Y = np.meshgrid(np.linspace(-aperture/2, aperture/2, n_grid),
                           np.linspace(-aperture/2, aperture/2, n_grid))
        rf += [np.stack([X.flatten(), Y.flatten(), np.full(fill_value=s, shape=X.flatten().shape)], axis=1),]
    s = 10.0
    n_grid = 1;
    X, Y = X, Y = np.meshgrid(np.linspace(0,0, n_grid),
                           np.linspace(0,0, n_grid))
    rf += [np.stack([X.flatten(), Y.flatten(), np.full(fill_value=s, shape=X.flatten().shape)], axis=1),]
    return np.concatenate(rf, axis=0)

def gauss_2d(center, sd, patch_size, orient_deg=0, aperture=1.0, dtype=np.float32):
    """
     Making a little gaussian blob. Can be elongated in one direction relative to other.
     [sd] is the x and y standard devs, respectively. 
     center and size are scaled according to the patch size, so that the blob always 
     has the same center/size relative to image even when patch size is different.
     aperture defines the number of arbitrary "units" occupied by the whole image
     units occupied by each pixel = aperture/patch_size.
        
     """
    if (not hasattr(sd,'__len__')) or len(sd)==1:
        sd = np.array([sd, sd])
        
    aspect_ratio = sd[0] / sd[1]
    orient_rad = orient_deg/180*np.pi
    
    # first meshgrid over image space
    x,y = np.meshgrid(np.linspace(-aperture/2, aperture/2, patch_size), \
                      np.linspace(-aperture/2, aperture/2, patch_size))
    
    new_center = copy.deepcopy(center) # make sure we don't edit the input value by accident!
    new_center[1] = (-1)*new_center[1] # negate the y coord so the grid matches w my other code
    
    x_centered = x-new_center[0]
    y_centered = y-new_center[1]
    
    # rotate the axes to match desired orientation (if orient=0, this is just regular x and y)
    x_prime = x_centered * np.cos(orient_rad) + y_centered * np.sin(orient_rad)
    y_prime = y_centered * np.cos(orient_rad) - x_centered * np.sin(orient_rad)

    # make my gaussian w the desired size/eccentricity
    gauss = np.exp(-((x_prime)**2 + aspect_ratio**2 * (y_prime)**2)/(2*sd[0]**2))
    
    # normalize so it will sum to 1
    gauss = gauss/np.sum(gauss)
    
    gauss = gauss.astype(dtype)
    
    return gauss

def get_prf_mask(center, sd, patch_size):
    
    """
    Get boolean mask for each pRF (region +/- 2 sds from center)
    To be used for getting overlap with coco semantic segmentations.
    """
    
    if np.all(np.abs(center)<0.50):
        
        # if the center of the pRF is within the image region, 
        # then can get max value without padding.
        prf = gauss_2d(center, sd, patch_size, aperture=1.0)
        # Creating a mask 2 SD from the center
        # cutoff of 0.14 approximates +/-2 SDs
        prf_mask = prf/np.max(prf)>0.14
        
    else:
        
        # otherwise need to pad array a little so that the center 
        # (max) will be included in the image.
        grid_space = 1.0/(patch_size-1)
        spaces_pad = int(np.ceil(0.5/grid_space))
        padded_aperture = 1.0+grid_space*spaces_pad*2
        padded_size = patch_size+spaces_pad*2
        prf_padded = gauss_2d(center, sd, patch_size=padded_size, \
                                 aperture=padded_aperture)
        # Creating a mask 2 SD from the center
        # cutoff of 0.14 approximates +/-2 SDs
        prf_mask_padded = prf_padded/np.max(prf_padded)>0.14
        # now un-pad it back to original size.
        prf_mask = prf_mask_padded[spaces_pad:spaces_pad+patch_size, \
                                   spaces_pad:spaces_pad+patch_size]
        
    return prf_mask


def pol_to_cart(angle_deg, eccen_deg):
    """
    Convert from polar angle coordinates (angle, eccentricity)
    to cartesian coordinates (x,y)
    Inputs and outputs in units of degrees.
    """
    angle_rad = angle_deg*np.pi/180
    x_deg = eccen_deg*np.cos(angle_rad)
    y_deg = eccen_deg*np.sin(angle_rad)
    
    return x_deg, y_deg

def cart_to_pol(x_deg, y_deg):
    """
    Convert from cartesian coordinates (x,y)
    to polar angle coordinates (angle, eccentricity)
    Inputs and outputs in units of degrees.
    """
    x_rad = x_deg/180*np.pi
    y_rad = y_deg/180*np.pi
    angle_rad = np.mod(np.arctan2(y_rad,x_rad), 2*np.pi)
    angle_deg = angle_rad*180/np.pi
    
    eccen_deg = np.sqrt(x_deg**2+y_deg**2)
    
    return angle_deg, eccen_deg
    