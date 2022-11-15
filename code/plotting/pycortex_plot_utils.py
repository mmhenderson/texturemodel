import os
import numpy as np
import copy
import cortex
import PIL.Image 

def maps_to_volumes(subject, maps, names, cmaps=None, mins=None, maxes=None, \
                    voxel_mask=None, xfmname=None, nii_shape=None, mask_3d = None, \
                   vox2plot=None):
    """
    Convert a list of maps into a dictionary of cortex.Volume objects
    maps should each be [n_voxels,] in shape where the mapping from n_voxels to 
    whole brain is given by voxel_mask.
    """
    
    substr = 'subj%02d'%subject
    
    if xfmname is None:
        xfmname = 'func1pt8_to_anat0pt8_autoFSbbr'

    volumes = {}
    
    if mins is None:
        mins = [None for ll in range(len(names))]
    elif len(mins)==1:
        mins = [mins[0] for ll in range(len(names))]
    if maxes is None:
        maxes = [None for ll in range(len(names))]
    elif len(maxes)==1:
        maxes = [maxes[0] for ll in range(len(names))]
    if cmaps is None:
        cmaps = ['PuBu' for ll in range(len(names))]
    elif len(cmaps)==1:
        cmaps = [cmaps[0] for ll in range(len(names))]  
    if vox2plot is None:
        vox2plot = [None for ll in range(len(names))]
    elif isinstance(vox2plot, np.ndarray):
        vox2plot = [vox2plot for ll in range(len(names))]
    else:
        assert(len(vox2plot)==len(maps))
    for ni, name in enumerate(names):
        
        map_thresh = copy.deepcopy(maps[ni])

        if vox2plot[ni] is not None: 
            map_thresh[~vox2plot[ni]] = np.nan

        volumes[name] = cortex.Volume(data = get_full_volume(map_thresh, voxel_mask, nii_shape), \
                                           cmap=cmaps[ni], subject=substr, \
                                           vmin=mins[ni], vmax=maxes[ni],\
                                           xfmname=xfmname, mask=mask_3d)
        
    return volumes

def plot_with_overlays(volumes, title, port, overlay_type='overlays', labels_on=True, recache=True):
    
    """
    Make sure to set recache=True if you are changing from one set of overlays to another 
    (or from no overlays to overlays)
    If plotting data from multiple subjects together, set recache=False
    Otherwise it will plot the first subject's overlays on all subject, which is wrong
    """
    overlay_names=('rois','sulci')

    pycortex_db_path = cortex.database.default_filestore
    subject_name = list(volumes.items())[0][1].subject
    overlay_file = os.path.join(pycortex_db_path, subject_name, '%s.svg'%overlay_type)
    print('using overlays from %s'%overlay_file)
    
    if labels_on:
        labels_visible=overlay_names
    else:
        labels_visible=()
    
    print('navigate browser to: 127.0.0.1:%s'%port)
    viewer = cortex.webshow(volumes, open_browser=True, port=port, \
                            autoclose=True, \
                            overlays_visible=overlay_names, \
                            overlays_available=overlay_names, \
                            labels_visible=labels_visible,\
                            overlay_file = overlay_file, \
                            recache=recache, 
                            title = title)
    
    return viewer

def get_roi_maps_for_pycortex(subject, roi_def):    
    """
    Create a dictionary of cortex.Volume labels for ROIs, to be plotted in PyCortex.
    
    subject: nsd subject number, 1-8
    roi_def: an roi definition object from utils.roi_utils.nsd_roi_def()
     
    """ 
    substr = 'subj%02d'%subject
    
    retlabs = roi_def.retlabs 
    has_ret = retlabs>-1
    facelabs = roi_def.facelabs 
    has_face = facelabs>-1
    placelabs = roi_def.placelabs
    has_place=placelabs>-1
    bodylabs = roi_def.bodylabs 
    has_body = bodylabs>-1
    
    retlabs[~has_ret] = np.nan
    facelabs[~has_face] = np.nan
    placelabs[~has_place] = np.nan
    bodylabs[~has_body] = np.nan

    voxel_mask = roi_def.voxel_mask
    nii_shape = roi_def.nii_shape
    xfmname = 'func1pt8_to_anat0pt8_autoFSbbr'
    mask_3d = np.reshape(voxel_mask, nii_shape, order='C')
        
    names = ['S%d ROI labels (retinotopic)'%subject, \
             'S%d ROI labels (face-selective)'%subject,\
             'S%d ROI labels (place-selective)'%subject,\
             'S%d ROI labels (body-selective)'%subject]
    maps = [retlabs, facelabs, placelabs, bodylabs]
    mins = [0,0,0,0]
    maxes = [np.max(m[~np.isnan(m)])+1 for m in maps]
    cmaps = ['Accent' for m in maps]
    
    roi_volumes = maps_to_volumes(subject, maps, names, cmaps=cmaps, mins=mins, maxes=maxes, \
                        voxel_mask=voxel_mask, nii_shape=nii_shape, mask_3d=mask_3d)
            
    return roi_volumes


def get_full_surface(values, voxel_mask):
    """
    For PyCortex: Put values for voxels that were analyzed back into their 
    correct coordinates in full surface space matrix.
    """
    full_vals = copy.deepcopy(voxel_mask).astype('float64')
    full_vals[voxel_mask==0] = np.nan
    full_vals[voxel_mask==1] = values
    
    return full_vals

def get_full_volume(values, voxel_mask, shape):
    """
    For PyCortex: Put values for voxels that were analyzed back into their 
    correct coordinates in full volume space matrix.
    """
    voxel_mask_3d = np.reshape(voxel_mask, shape)
    full_vals = copy.deepcopy(voxel_mask_3d).astype('float64')
    full_vals[voxel_mask_3d==0] = np.nan
    full_vals[voxel_mask_3d==1] = values
    
    full_vals = np.moveaxis(full_vals, [0,1,2], [2,1,0])
    
    return full_vals


def crop_image(fns, fns_cropped, bbox_new=[600,900,2300,2500]):
    """
    Utility to crop images with a specified bounding box
    Using this to crop my flatmap images to a specified size
    """
    if isinstance(fns, str):
        fns = [fns]
    if isinstance(fns_cropped, str):
        fns_cropped = [fns_cropped]
    
    for fn, fn_cropped in zip(fns, fns_cropped):
        
        im = PIL.Image.open(fn)
        
        im_cropped = im.crop(bbox_new)
        
        im_cropped.save(fn_cropped)