import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import os
import copy

from utils import roi_utils, nsd_utils
from plotting import plot_utils, pycortex_plot_utils, plot_prf_params

"""
General use functions for plotting encoding model fit results.
Input to most of these functions is 'out', which is a dictionary containing 
encoding model fit results, created by the model fitting code in 
model_fitting/fit_model.py

"""

def get_substr(out):   
    if hasattr(out, 'keys'):        
        # single subject case
        substr = 'S%d'%out['subject']        
    else:        
        # multi subject case   
        if len(out)>1:
            substr = 'combined '+', '.join(['S%d'%o['subject'] for o in out])  
        else: 
            substr = 'S%d'%(out[0]['subject'])
            
    return substr

def get_noise_ceiling(out):
    
    noise_ceiling = nsd_utils.get_nc(subject=out['subject'], \
                                     average_image_reps = out['average_image_reps']) 
    return noise_ceiling

def barplot_R2_all(fitting_type, out, roi_def, ylims = [-0.05, 0.30], \
                   nc_thresh=0.01):
    
    n_subjects = len(out)
    n_rois = roi_def.n_rois
    roi_names = roi_def.roi_names

    vals = np.zeros((n_subjects, n_rois, 1))

    for si in range(n_subjects):

        val_r2 = out[si]['val_r2']  
        nc = get_noise_ceiling(out[si])
        inds2use = nc>nc_thresh

        for ri in range(n_rois):

            inds_this_roi = roi_def.ss_roi_defs[si].get_indices(ri) & inds2use
            assert(np.sum(inds_this_roi)>0)
            vals[si,ri,:] = np.mean(val_r2[inds_this_roi,0], axis=0)

    mean_vals = np.mean(vals, axis=0)
    sem_vals = np.std(vals, axis=0) / np.sqrt(n_subjects)

    title='%s\n%s\nShowing all voxels with noise ceiling R2>%.2f'%(get_substr(out), \
                                                                           fitting_type, nc_thresh)

    plot_utils.set_all_font_sizes(fs = 16)
    fh = plot_utils.plot_multi_bars(mean_data=mean_vals, err_data=sem_vals, colors=np.array([[0.8, 0.8, 0.8]]), space=0.2, \
                    xticklabels=roi_names, ylabel='R2', \
                    ylim=ylims, title=title, horizontal_line_pos=0,\
                    legend_labels=None, \
                    legend_overlaid=False, legend_separate=False, \
                    fig_size=(16,4))

    # now adding single subjects to the plot too
    subcolors = cm.viridis(np.linspace(0,1,n_subjects))
    for ss in range(n_subjects):
        plt.plot(np.arange(n_rois), vals[ss,:,0],'.',markersize=10, markeredgecolor='none', \
                 markerfacecolor=subcolors[ss,:], zorder=15)
     
    lh = plt.figure();
    for ss in range(n_subjects):
        plt.plot(0,ss,'.',markersize=10, markeredgecolor='none', \
                 markerfacecolor=subcolors[ss,:], )
    plt.legend(['S%d'%(ss+1) for ss in range(n_subjects)])
    
    return fh, lh

    
def plot_perf_summary(fitting_type, out, fig_save_folder=None):
    """
    Plot some general metrics of fit performance, across all voxels.
    """
    
    if hasattr(out, 'keys'):        
        # single subject case
        best_losses = out['best_losses'][:,0]
        val_cc = out['val_cc'][:,0]
        val_r2 = out['val_r2'][:,0]
        best_lambdas = out['best_lambdas']
        lambdas = out['lambdas']        
    else:        
        # multi subject case, concat all voxels
        best_losses =np.concatenate([o['best_losses'][:,0] for o in out], axis=0)
        val_cc =np.concatenate([o['val_cc'][:,0] for o in out], axis=0)
        val_r2 =np.concatenate([o['val_r2'][:,0] for o in out], axis=0)
        best_lambdas =np.concatenate([o['best_lambdas'][:,0] for o in out], axis=0)
        lambdas = out[0]['lambdas']
        assert(np.all([np.all(lambdas==o['lambdas']) for o in out]))
        
    plt.figure(figsize=(16,8));

    plt.subplot(2,2,1)
    plt.hist(best_losses,100)
    plt.xlabel('loss value/SSE (held-out training)');
    plt.ylabel('number of voxels');

    plt.subplot(2,2,2)
    plt.hist(val_cc,100)
    cclims = [-1,1]    
    plt.xlim(cclims)
    plt.xlabel('correlation coefficient r (validation)');
    plt.ylabel('number of voxels');
    plt.axvline(0,color='k')

    plt.subplot(2,2,3)
    plt.hist(val_r2,100)
    plt.xlabel('r2 (validation)');
    plt.ylabel('number of voxels');
    plt.axvline(0,color='k')

    plt.subplot(2,2,4)
    plt.plot(lambdas, [np.sum(best_lambdas==k) for k in range(len(lambdas))], lw=4, marker='o', ms=12)
    plt.xscale('log');
    plt.xlabel('lambda value for best fit');
    plt.ylabel('number of voxels');

    plt.suptitle('%s\n%s'%(get_substr(out), fitting_type))
    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'fit_summary.png'))
        plt.savefig(os.path.join(fig_save_folder,'fit_summary.pdf'))


def plot_summary_pycortex(fitting_type, out, port, roi_def=None, simplest_roi_maps=False):
    
    """
    Use pycortex webgl function to plot some summary statistics for encoding model fits, in surface space.
    Plots pRF spatial parameters, and the model's overall prediction performance on validation set.
    If roi_def is included, then will also plot ROI masks.
    """
    
    if not hasattr(out, 'keys'):
        # multi subject case
        subjects = [o['subject'] for o in out]
    else:
        # single subject case 
        subjects = [out['subject']]
        out = [copy.deepcopy(out)]
        
    names_each = ['pRF eccentricity', 'pRF angle','pRF size', \
              'Correlation (validation set)','R2 (validation set)']
    mins_each = [0, 0, 0, 0, 0]
    maxes_each = [9, 360, 9, 0.8, 0.6]
    cmaps_each = ['PRGn', 'Retinotopy_RYBCR', 'PRGn', 'PuBu','PuBu']

    title = '%s, %s'%(get_substr(out), fitting_type);

    vox2plot = []
    maps = []
    cmaps = []
    mins = []
    maxes = []
    subject_map_inds = []
    names = []

    for si, ss in enumerate(subjects):

        best_ecc_deg, best_angle_deg, best_size_deg = \
                    plot_prf_params.get_prf_pars_deg(out[si], screen_eccen_deg=8.4)
        val_cc = out[si]['val_cc'][:,0]
        val_r2 = out[si]['val_r2'][:,0]
        maps += [best_ecc_deg, best_angle_deg, best_size_deg, val_cc, val_r2]
        
        vox2plot.append(val_r2>0.01)
        
        names += ['S%d: %s'%(ss, name) for name in names_each]
        cmaps += cmaps_each
        mins += mins_each
        maxes += maxes_each
        subject_map_inds += [si for name in names_each]

    voxel_mask = [o['voxel_mask'] for o in out]
    nii_shape = [o['brain_nii_shape'] for o in out]
    volume_space = out[0]['volume_space']
    
    pycortex_plot_utils.plot_maps_pycortex(subjects, port, maps, names, \
                            subject_map_inds=subject_map_inds, \
                            mins=mins, maxes=maxes, cmaps=cmaps, \
                            title=title, vox2plot = vox2plot, roi_def=roi_def, \
                            voxel_mask =voxel_mask, \
                            nii_shape = nii_shape, \
                            volume_space=volume_space, \
                            simplest_roi_maps=simplest_roi_maps)

def plot_fit_summary_volume_space(fitting_type, out, roi_def=None, screen_eccen_deg = 8.4, \
                                fig_save_folder=None):
    """
    Visualize some basic properties of pRFs for each voxel, in volume space
    Makes plots for different cross sections of the brain as 2D images. Not as nice
    as the pycortex plots, but is a good sanity check for brain mask/retinotopy.
    """     
    
    if not hasattr(out, 'keys'):
        raise ValueError('can only use this for single-subject data presently.')
      
    if out['brain_nii_shape'] is None:
        raise ValueError('Cannot use this function for data that is in surface space, should use pycortex to visualize instead.')

    brain_nii_shape = out['brain_nii_shape']
    voxel_idx = out['voxel_index'][0]
    
    best_losses = out['best_losses'][:,0]
    val_cc = out['val_cc'][:,0]
    best_ecc_deg, best_angle_deg, best_size_deg = plot_prf_params.get_prf_pars_deg(out, screen_eccen_deg)    
    
    if roi_def is not None:
        retlabs = roi_def.retlabs
    else:
        retlabs = out['voxel_roi'][0]
        
    volume_loss = roi_utils.view_data(brain_nii_shape, voxel_idx, best_losses)
    volume_cc   = roi_utils.view_data(brain_nii_shape, voxel_idx, val_cc)
    volume_ecc  = roi_utils.view_data(brain_nii_shape, voxel_idx, best_ecc_deg)
    volume_ang  = roi_utils.view_data(brain_nii_shape, voxel_idx, best_angle_deg)
    volume_size = roi_utils.view_data(brain_nii_shape, voxel_idx, best_size_deg)
    volume_roi = roi_utils.view_data(brain_nii_shape, voxel_idx, retlabs)

    slice_idx = 40
    fig = plt.figure(figsize=(16,8))
    plt.subplot(2,3,1)
    plt.title('Loss')
    plt.imshow(volume_loss[:,:,slice_idx], cmap='viridis', interpolation='None')
    plt.colorbar()
    plt.subplot(2,3,2)
    plt.title('Validation accuracy')
    plt.imshow(volume_cc[:,:,slice_idx], cmap='viridis', interpolation='None')
    plt.colorbar()
    plt.subplot(2,3,3)
    plt.title('RF Eccentricity')
    plt.imshow(volume_ecc[:,:,slice_idx], cmap='viridis', interpolation='None')
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.title('RF Angle')
    plt.imshow(volume_ang[:,:,slice_idx], cmap='hsv', interpolation='None')
    plt.colorbar()
    plt.subplot(2,3,5)
    plt.title('RF Size')
    plt.imshow(volume_size[:,:,slice_idx], cmap='viridis', interpolation='None')
    plt.colorbar()
    plt.subplot(2,3,6)
    plt.title('ROI labels')
    plt.imshow(volume_roi[:,:,slice_idx], cmap='jet', interpolation='None')
    plt.colorbar()

    plt.suptitle('S%02d, %s'%(out['subject'], fitting_type));
    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'fit_summary_volumespace.pdf'))
        plt.savefig(os.path.join(fig_save_folder,'fit_summary_volumespace.png'))
        
def plot_noise_ceilings(fitting_type,out, fig_save_folder=None):
    """
    Plot distribution of noise ceilings and NCSNR across all voxels.
    This is just a property of the voxels; should be same for any encoding model fit.
    """    
    
    if hasattr(out, 'keys'):        
        # single subject case
        noise_ceiling = get_noise_ceiling(out)
        voxel_ncsnr = out['voxel_ncsnr'][out['voxel_index']]
    else:       
        # multi subject case, concat all voxels
        noise_ceiling = np.concatenate([get_noise_ceiling(o) for o in out], axis=0)
        voxel_ncsnr = np.concatenate([o['voxel_ncsnr'][o['voxel_index']] for o in out], axis=0)

    plt.figure(figsize=(16,4));

    plt.subplot(1,2,1)
    plt.hist(voxel_ncsnr,100)
    plt.xlabel('NCSNR');
    plt.ylabel('number of voxels');

    plt.subplot(1,2,2)
    plt.hist(noise_ceiling,100)
    plt.xlabel('Noise ceiling (percent)');
    plt.ylabel('number of voxels');
    plt.axvline(0,color='k')

    plt.suptitle('%s\n%s'%(get_substr(out), fitting_type))
    
    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'noise_ceiling_dist.png'))
        plt.savefig(os.path.join(fig_save_folder,'noise_ceiling_dist.pdf'))

def plot_cc_each_roi(fitting_type, out, roi_def, skip_inds=None, fig_save_folder=None, fig_size=None):
    """
    Make a histogram for each ROI, showing distibution of validation set correlation coefficient for all voxels.
    """
    
    n_rois = roi_def.n_rois
    roi_names = roi_def.roi_names
    
    if skip_inds is None:
        skip_inds = []
    
    if hasattr(out, 'keys'):
        # single subject case
        val_cc = out['val_cc'][:,0]
    else:
        # multi subject case, concatenate
        val_cc = np.concatenate([o['val_cc'][:,0] for o in out], axis=0)
        
    if fig_size is None:
        fig_size = (16,12)
    plt.figure(figsize=fig_size)
    npy = int(np.ceil(np.sqrt(n_rois-len(skip_inds))))
    npx = int(np.ceil((n_rois-len(skip_inds))/npy))
    pi2label = [(npx-1)*npy+1]
    
    pi=0
    for rr in range(n_rois):
        
        if rr not in skip_inds:
            
            inds_this_roi = roi_def.get_indices(rr)
            
            pi+=1
            plt.subplot(npx,npy,pi)
            h = plt.hist(val_cc[inds_this_roi], bins=np.linspace(-0.2,1,100))

            if pi in pi2label:
                plt.xlabel('Correlation coefficient')
                plt.ylabel('Number of voxels')
            else:
                plt.xticks([]);

            plt.axvline(0,color='k')
            plt.title('%s (%d vox)'%(roi_names[rr], np.sum(inds_this_roi)))

    plt.suptitle('Correlation coef. on validation set\n%s\n%s'%(get_substr(out), fitting_type));

    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'corr_each_roi.pdf'))
        plt.savefig(os.path.join(fig_save_folder,'corr_each_roi.png'))
             
        
def plot_r2_vs_nc(fitting_type, out, roi_def, skip_inds=None, \
                  axlims=None, fig_save_folder=None, fig_size=None, \
                  sub_colors=None):

    """
    Create scatter plots for each ROI, comparing each voxel's R2 prediction to the noise ceiling.
    """
  
    if hasattr(out, 'keys'):
        noise_ceiling = get_noise_ceiling(out)
        val_r2 = out['val_r2'][:,0]       
        n_subs=1
        n_vox_each = [val_r2.shape[0]]
    else:
        # multi subject case, concat all voxels
        noise_ceiling = np.concatenate([get_noise_ceiling(o) for o in out], axis=0)
        # multi subject case, concat all voxels
        val_r2 = np.concatenate([o['val_r2'][:,0] for o in out], axis=0)
        # color diff subjects differently
        n_subs = len(out)
        n_vox_each = np.array([o['val_r2'].shape[0] for o in out])
    
    group_color_inds = np.repeat(np.arange(n_subs), n_vox_each)
    
    if sub_colors is not None:
        assert(sub_colors.shape[0]==n_subs)
    else:
#         sub_colors = cm.Set2(np.linspace(0,1,n_subs))
        colors = cm.tab10(np.linspace(0,1,n_subs))
#         sub_colors[:,3] = 0.1 # make each set of points transparent
 
    dat2plot = np.concatenate([noise_ceiling[:,np.newaxis],val_r2[:,np.newaxis]], axis=1)
    suptitle = '%s\n%s\nComparing model performance to noise ceiling'\
                           %(get_substr(out), fitting_type)
    
    inds2use = np.ones(np.shape(val_r2))==1
    if axlims is None:
        axlims = [-0.1, 1.1]

    sp = plot_utils.scatter_plot(color=sub_colors, xlabel='Noise Ceiling', ylabel='R2', \
                                 xlims=axlims, ylims=axlims, \
                                 xticks=[0, 0.5, 1.0], yticks=[0, 0.5, 1.0],\
                                 show_diagonal=True, show_axes=True);
    if fig_size is None:
        fig_size = (16,16)

    plot_utils.create_roi_subplots(dat2plot, inds2use, sp, roi_def, \
                                   subject_inds=group_color_inds, \
                                   skip_inds=skip_inds, \
                                   suptitle=suptitle, \
                                   label_just_corner=True, \
                                   figsize=fig_size)

    plt.gcf().subplots_adjust(bottom=0.5)
    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'r2_vs_noiseceiling.png'))
        plt.savefig(os.path.join(fig_save_folder,'r2_vs_noiseceiling.pdf'))
          