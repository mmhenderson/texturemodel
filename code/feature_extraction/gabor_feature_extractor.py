import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Create a pytorch module that performs image filtering with Gabor filters at various
orientations and spatial frequencies. 
Different spatial frequencies are achieved by resampling the input to different sizes.

Original source of this code is the github repository:
https://github.com/styvesg/nsd
It was modified by MH to work with the fitting pipeline for this project.

"""



class gabor_extractor_multi_scale(nn.Module):
    
    """
    Module to extract maps of orientation content of images at various spatial scales.
    Input parameters:
        n_ori                  ~ number of linearly spaced orientations in [0,pi]
        n_sf                   ~ how many spatial frequencies (scales) to use?
        sf_range_cyc_per_stim  ~ min and max spatial frequency, units of cycles/stimulus.
        log_spacing            ~ do you want to log space the SF values? if false, use linear.
        pix_per_cycle          ~ how many pixels will be used to depict one cycle. default = 2, i.e., the Nyquist limit.
                                 If too low, will result in jaggy edges or aliasing, but if too high, usually will require 
                                 stimuli with larger than native resolution.
        cycles_per_radius      ~ determines radius of gaussian envelop. we specify how many cycles the sinewave completes
                                 per radius (1std.) of the gaussian envelope. default = 1 = one cycle of the sinewave per 
                                 std. of the gaussian envelope.
        radii_per_filter       ~ determines the size of the filter. we specify how many radii (1std. each) of the gaussian
                                 envelope fits inside of the filter. default = 4.
        complex_cell           ~ default = True, meaning that ea. feature map represents a given spatial frequency 
                                 that is phase invariant (the absolute value is taken between a pair of feature maps 
                                 constructed using filters with 0 and pi/2 phase). if False, we distinguish between 
                                 filters with 0 and pi/2 phase, resulting in 2 feature maps for each spatial frequency.
        padding_mode           ~ padding mode parameter passed to the torch.nn.Conv2d function. Default circular. 
                                 Different modes (such as constant) can create edge artifacts.
        RGB                    ~ True for color images with 3 channels (RGB), False for grayscale with 1 color channel.
        device                 ~ Device that the module's parameters will be on (i.e. cpu or cuda)
    Returns a nn.Module into which images can be directly passed.
              
    """
    
    
    def __init__(self, n_ori=4, n_sf=4, sf_range_cyc_per_stim = (3, 72), log_spacing = True,
                 pix_per_cycle=4.13, cycles_per_radius=0.7, 
                 radii_per_filter=4, complex_cell=True, padding_mode = 'circular', RGB=False, 
                 nonlin_fn=None, device = None):
    
        super(gabor_extractor_multi_scale, self).__init__()
        
        self.n_ori = n_ori
        self.n_sf = n_sf
        self.sf_range_cyc_per_stim = sf_range_cyc_per_stim
        self.log_spacing = log_spacing
        self.pix_per_cycle = pix_per_cycle
        self.cycles_per_radius = cycles_per_radius
        self.radii_per_filter = radii_per_filter          
        self.complex_cell = complex_cell
        self.padding_mode = padding_mode
        self.nonlin_fn = nonlin_fn
        if self.nonlin_fn is not None:
            print('adding nonlinearity function:')
            print(self.nonlin_fn)
        self.RGB = RGB # NOTE THE CODE HASN'T BEEN TESTED WITH RGB=TRUE
        if device is None:
            device = 'cpu:0'
        self.device = device
        
        self.compute_filter_pars()
        self.make_param_grid()
        self.make_filter_stack()
        self.make_submodules()
        
    def compute_filter_pars(self):
        
        """
        Compute some basic parameters of the gabor filters to use here
        """
    
        # Radius (1 std) of gaussian envelope of gabor filters, in pixels
        self.envelope_radius_pix = self.pix_per_cycle * self.cycles_per_radius

        # Cycles per filter
        self.cycles_per_filter = self.cycles_per_radius * self.radii_per_filter ##should be constant

        # Given the num of cycles in a filter and the pix/cyc, this is how big the filter should be in pixels
        # This is the same for all filter SFs - the image is just resized to different scales.
        self.pix_per_filter = int(np.round(self.cycles_per_filter * self.pix_per_cycle)) ##should be constant

        # Going to put all these filter pars into a dict for easy saving later
        self.gabor_filter_pars={'pix_per_cycle': self.pix_per_cycle, 'cycles_per_radius': self.cycles_per_radius, \
                                'radii_per_filter': self.radii_per_filter, 'cycles_per_filter': self.cycles_per_filter, \
                                'pix_per_filter': self.pix_per_filter, 'envelope_radius_pix': self.envelope_radius_pix}
                     
    def make_param_grid(self):
        
        """
        Create a grid sampling each desired orientation and spatial frequency.
        """
        
        self.orients_rad_unique = np.linspace(0, np.pi, num=self.n_ori+1)[:-1]

        if self.log_spacing:
            sfs_cyc_per_stim = np.logspace(np.log10(self.sf_range_cyc_per_stim[0]), np.log10(self.sf_range_cyc_per_stim[1]),num = self.n_sf)
        else:
            sfs_cyc_per_stim = np.linspace(self.sf_range_cyc_per_stim[0], self.sf_range_cyc_per_stim[1], num = self.n_sf)

                         
        # What are the sizes the images must be resampled to, to achieve filtering at each desired scale?
        self.sizes_to_resample_each_scale = np.round(sfs_cyc_per_stim * self.pix_per_cycle).astype('int')
        # after the rounding step above, these are the actual frequencies (very close to input)
        sfs_cyc_per_stim_actual = self.sizes_to_resample_each_scale/self.pix_per_cycle
        
        if not self.complex_cell:
            self.n_phases = 2
            phases = [0, np.pi/2]
        else:
            self.n_phases = 1
            phases = [0]

        # Creating a grid of all the sfs/orients we would like to sample        
        sf_list = np.repeat(sfs_cyc_per_stim, self.n_ori*self.n_phases)
        sf_list_actual = np.repeat(sfs_cyc_per_stim_actual, self.n_ori*self.n_phases)
        self.size_list = np.repeat(self.sizes_to_resample_each_scale, self.n_ori*self.n_phases)
        self.ori_list_rad = np.tile(np.repeat(self.orients_rad_unique, self.n_phases), self.n_sf)
        ori_list_deg = self.ori_list_rad*180/np.pi
        self.phase_list_rad = np.tile(phases, self.n_ori*self.n_sf)

        # Putting these into a dataframe format for easy saving later
        # The order of this corresponds to the order of the features that are extracted 
        # (lowest SF to highest, with orientations and phases listed within each SF)
        self.feature_table = pd.DataFrame.from_dict({'SF: cycles per stim': sf_list, \
                                               'SF: cycles per stim (actual)': sf_list_actual, \
                                               'Size in pixels when filtered': self.size_list, \
                                               'Orientation: radians': self.ori_list_rad, \
                                               'Orientation: degrees': ori_list_deg, \
                                               'Phase of filter (if simple cells)': self.phase_list_rad})
        
        # since the filters are actually same at all spatial phases, we only need to create the 
        # first n_ori*n_phases of these filters. the rest would be identical.
        self.n_unique_filters = self.n_ori*self.n_phases
        self.n_features = self.n_ori*self.n_sf*self.n_phases
    
    def make_filter_stack(self):
        
        """
        Create a stack of Gabor filters to be used by this module.
        Note this same stack is recycled and used again at each spatial scale, because the images are resized differently.
        """
        
        if self.RGB==True:
            self.color_channels=3
        else:
            self.color_channels=1
        self.filter_stack = np.zeros((self.n_unique_filters, self.color_channels, self.pix_per_filter, self.pix_per_filter))
        if self.complex_cell:
            self.filter_stack = self.filter_stack+1j

        center = (0,0)
        freq = self.cycles_per_filter
        radius = np.float32(self.envelope_radius_pix)
        for ii in range(self.n_unique_filters):
            ori = self.ori_list_rad[ii]
            for cc in range(self.color_channels):
                if self.complex_cell:
                    self.filter_stack[ii,cc,:,:] = make_complex_gabor(freq,ori,center,radius,self.pix_per_filter)
                else:
                    ph = self.phase_list_rad[ii]
                    self.filter_stack[ii,cc,:,:] = make_gabor(freq,ori,ph,center,radius,self.pix_per_filter)

        #split into real and imag gabor filters and represent ea. as tensor 
        if self.complex_cell:
            self.real_filters_tnsr = nn.Parameter(torch.tensor(np.real(self.filter_stack), dtype=torch.float32, requires_grad=True).to(self.device))
            self.imag_filters_tnsr = nn.Parameter(torch.tensor(np.imag(self.filter_stack), dtype=torch.float32, requires_grad=True).to(self.device))
        else:
            self.real_filters_tnsr = nn.Parameter(torch.tensor(self.filter_stack, dtype=torch.float32, requires_grad=True).to(self.device))
            self.imag_filters_tnsr = None
            
    def make_submodules(self):
        
        """
        Make a list of feature extractors, ea. extracting a diff freq (by resampling image to diff size before conv)
        Each submodule is type 'gabor_extractor_single_scale'
        """
        
        self.gfe_list = []
        
        for resam_size in self.sizes_to_resample_each_scale:
            
            # Create instance of torch feature extractor with the given resampling parameter
            feature_extractor = gabor_extractor_single_scale(self.real_filters_tnsr, self.imag_filters_tnsr, 
                                                   (resam_size, resam_size), padding_mode=self.padding_mode)
            self.gfe_list.append(feature_extractor)
            
    def forward(self, image_batch):
        """
        Process a set of images with filters at every orientation and spatial frequency in the bank.
        Inputs:
            size [n_images x n_channels x height x width]
            Where n_channels has to be 1 (this code won't work for RGB currently.)
        Returns:
            a list of feature map stacks (one for each spatial scale).
            Each stack goes [n_images x n_ori x height x width]
       
        """
       
        feature_map_list = []

        for gfe in self.gfe_list:
           
            #for given freq, create feature map for each gabor orientation 
            sing_freq_features = gfe(image_batch)   #size [num stim, num orientations, stim pix, stim pix]
            
            if self.nonlin_fn is not None:
                sing_freq_features = self.nonlin_fn(sing_freq_features)
                
            #put each feature map tensor into a list of tensors 
            feature_map_list.append(sing_freq_features)

        return feature_map_list 
    
    
    def get_fmaps_sizes(self, image_size):
        """ 
        Compute sizes of the feature maps that would be returned by this module, for a given input size.
        Doesn't actually compute any real features.
        Returns number of total features across all groups of maps, and the resolution of each map group.
        """
        n_features = 0
        _x_fake = torch.zeros(size=[1,1, image_size[0], image_size[1]]).to(self.device)
        _fmaps = self.forward(_x_fake)
        resolutions_each_sf = []
        for k,_fm in enumerate(_fmaps):
            n_features = n_features + _fm.size()[1]
            resolutions_each_sf.append(_fm.size()[2])

        assert(self.n_features==n_features)
        self.resolutions_each_sf = resolutions_each_sf
        
      
class gabor_extractor_single_scale(nn.Module):    
    
    """ 
    Apply a stack of Gabor filters to images, at a given scale.
    Return stack of feature maps.
    """
    
    def __init__(self, real_filters_tnsr, imag_filters_tnsr, new_dim, padding_mode='circular'):
        super(gabor_extractor_single_scale, self).__init__()
        
        self.real_filters_tnsr = real_filters_tnsr
        self.imag_filters_tnsr = imag_filters_tnsr
        self.padding_mode = padding_mode
       
        # This will be the stimulus resampling function 
        self.resam = torch.nn.Upsample(new_dim, mode="bilinear", align_corners=True)
      
    def forward(self, image_batch):
        
        # Resize stimuli
        resampled_stim_stack = self.resam(image_batch)
        input_sz = resampled_stim_stack.shape[2]

        # Determine padding size (half the filter width)
        pad_sz = int(np.floor(self.real_filters_tnsr.shape[2]/2))
        
        nsamples = image_batch.shape[0]
        nfilters = self.real_filters_tnsr.shape[0]
        filter_size = self.real_filters_tnsr.shape[2]
        device = image_batch.get_device()
        if device<0:
            device = image_batch.device

        # Convolve stim with filters (returns feat maps of size [num stim, num orientations, stim pix, stim pix])
        # creating a conv2d layer here because it allows custom padding mode (often want circular mode because it 
        # minimizes edge artifacts compared to zero-padding)
        c = torch.nn.Conv2d(nsamples, nfilters, filter_size, stride=1, padding=pad_sz, padding_mode=self.padding_mode).to(device)
        c.weight.data = self.real_filters_tnsr
        c.bias.data.fill_(0)
        real_feature_map_tnsr = c(resampled_stim_stack)    
        
        # If it was a complex_cell, get imag feature map as well and square/sum real and imag parts
        if self.imag_filters_tnsr is not None:

            c = torch.nn.Conv2d(nsamples, nfilters, filter_size, stride=1, padding=pad_sz, padding_mode=self.padding_mode).to(device)
            c.weight.data = self.imag_filters_tnsr
            c.bias.data.fill_(0)
            imag_feature_map_tnsr = c(resampled_stim_stack)
            
            fmap_batch = torch.sqrt((real_feature_map_tnsr**2)+(imag_feature_map_tnsr**2))
        else: 
            
            # Otherwise just stick with this map.
            fmap_batch = real_feature_map_tnsr
        
        return fmap_batch
        


# Support functions below here

# def add_nonlinearity(module, nonlinearity):
#     new_module = module
#     new_module.forward = nonlinearity(module.forward)
#     return new_module


# class add_nonlinearity(nn.Module):
#     def __init__(self, _fmaps_fn, _nonlinearity):
#         super(add_nonlinearity, self).__init__()
#         self.fmaps_fn = _fmaps_fn
#         self.nl_fn = _nonlinearity
#     def forward(self, _x):
#         return [self.nl_fn(_fm) for _fm in self.fmaps_fn(_x)]


def make_2D_sinewave(freq, theta, phase, n_pix):
    '''
    freq is cycles/image
    theta is in radians
    phase is in radians (0 pi)
    center is (x,y) in pixel coordinates
    n_pix is size of the kernel in pixels
    '''
    vec = np.array([np.cos(theta), np.sin(theta)])*2*np.pi*freq / n_pix
    
    [Xm, Ym] = np.meshgrid(np.linspace(-.5*n_pix, .5*n_pix, n_pix), np.linspace(-.5*n_pix, .5*n_pix, n_pix))
    
    proj = np.array([Xm.ravel(), Ym.ravel()]).T.dot(vec)
    
    Dt = np.sin(proj+phase)              # compute proportion of Xm for given orientation
    Dt = Dt.reshape(Xm.shape)
    
    return Dt

def make_gaussian(center, sig, n_pix):
    """
    Make a picture of a circular gaussian blob.
    center is the center of the blob in pixels. center of image is (0,0)
    sig is one std. of the gaussian (pixels)
    n_pix is the size of the picture of the gaussian blob. i.e., output will be an 2D array that is n_pix-by-n_pix
    """
    
    [Xm, Ym] = np.meshgrid(np.linspace(-.5*n_pix, .5*n_pix, n_pix), np.linspace(-.5*n_pix, .5*n_pix, n_pix))
    
    x0 = center[0]
    y0 = center[1]
    
    Z = (1. / (2*np.pi*sig**2))
    
    return Z *np.exp(-((Xm-x0)**2 + (Ym-y0)**2) / (2*sig**2))


def make_gabor(freq, theta, phase, center, sig, n_pix):
    return make_2D_sinewave(freq, theta, phase, n_pix) * make_gaussian(center, sig,n_pix)


def make_complex_gabor(freq, theta, center, sig, n_pix):
    '''
    make_complex_gabor(freq, theta, center, sig, n_pix)
    freq is spatial frequency in cycles/image
    theta is orientation in radians
    center is (x,y) in pixel coordinates. center of image is (0,0)
    sig is one std of the gaussian envelope (pixels)
    n_pix is size of the kernel in pixels
    
    '''
    phase = 0
    on_gabor = make_gabor(freq, theta, phase, center, sig, n_pix)
    phase = np.pi/2.
    off_gabor = make_gabor(freq, theta, phase, center, sig, n_pix)
    
    return off_gabor + 1j*on_gabor