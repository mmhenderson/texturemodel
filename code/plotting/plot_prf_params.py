import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

from plotting import summary_plots
    
"""
General use functions for plotting spatial pRFs from FWRF encoding model.
Input to most of these functions is 'out', which is a dictionary containing 
encoding model fit results, created by the model fitting code in 
model_fitting/fit_model.py

"""

def get_prf_pars_deg(out, screen_eccen_deg=8.4):
    """
    Convert the saved estimates of prf position/sd into eccentricity, angle, etc in degrees.
    """
    if len(out['best_params'])==7:
        best_models, weights, bias, features_mt, features_st, best_model_inds, _ = out['best_params']
    else:
        best_models, weights, bias, features_mt, features_st, best_model_inds = out['best_params']
    best_models_deg = best_models * screen_eccen_deg
    if len(best_models_deg.shape)==2:
        best_models_deg = np.expand_dims(best_models_deg, axis=1)
    pp=0
    best_ecc_deg  = np.sqrt(np.square(best_models_deg[:,pp,0]) + np.square(best_models_deg[:,pp,1]))
    best_angle_deg  = np.mod(np.arctan2(best_models_deg[:,pp,1], best_models_deg[:,pp,0])*180/np.pi, 360)
    best_angle_deg[best_ecc_deg==0.0] = np.nan
    best_size_deg = best_models_deg[:,pp,2]
    
    return best_ecc_deg, best_angle_deg, best_size_deg


def plot_spatial_rf_circles(fitting_type, out, roi_def, skip_inds=None, r2_cutoff = 0.10, \
                            screen_eccen_deg = 8.4, fig_save_folder = None):

    """
    Make a plot for each ROI showing the visual field coverage of pRFs in that ROI.
    Each circle is a voxel, with the size of the circle indicating the pRF's size (1 SD)
    """

    if hasattr(out, 'keys'):        
        # single subject case
        best_models_deg = out['best_params'][0] * screen_eccen_deg
        val_r2 = out['val_r2'][:,0]
    else:        
        # multi subject case   
        best_models_deg = np.concatenate([o['best_params'][0] * screen_eccen_deg for o in out], axis=0)
        val_r2 = np.concatenate([o['val_r2'][:,0] for o in out])
        
    pp=0
   
    n_rois = roi_def.n_rois
    roi_names = roi_def.roi_names
   
    if skip_inds is None:
        skip_inds = []
    
    abv_thresh = val_r2>r2_cutoff
    
    plt.figure(figsize=(24,18))
    npy = int(np.ceil(np.sqrt(n_rois-len(skip_inds))))
    npx = int(np.ceil((n_rois-len(skip_inds))/npy))

    pi2label = [(npx-1)*npy+1]
   
    pi = 0
    for rr in range(n_rois):
        
        if rr not in skip_inds:
            
            inds_this_roi = roi_def.get_indices(rr)

            inds2use = np.where(inds_this_roi & abv_thresh)[0]

            pi+=1
            plt.subplot(npx,npy,pi)
            ax = plt.gca()

            for vi, vidx in enumerate(inds2use):

                plt.plot(best_models_deg[vidx,pp,0], best_models_deg[vidx,pp,1],'.',color='k')
                circ = matplotlib.patches.Circle((best_models_deg[vidx,pp,0], \
                                                  best_models_deg[vidx,pp,1]), \
                                                  best_models_deg[vidx,pp,2], 
                                                  color = [0.8, 0.8, 0.8], fill=False)
                ax.add_artist(circ)

            plt.axis('square')

            plt.xlim([-1.5*screen_eccen_deg, 1.5*screen_eccen_deg])
            plt.ylim([-1.5*screen_eccen_deg, 1.5*screen_eccen_deg])
            plt.xticks(np.arange(-8,9,4))
            plt.yticks(np.arange(-8,9,4))

            boxcolor = [0.6, 0.6, 0.6]
            plt.plot([screen_eccen_deg/2,screen_eccen_deg/2], \
                     [screen_eccen_deg/2, -screen_eccen_deg/2],color=boxcolor)
            plt.plot([-screen_eccen_deg/2,-screen_eccen_deg/2], \
                     [screen_eccen_deg/2, -screen_eccen_deg/2],color=boxcolor)
            plt.plot([-screen_eccen_deg/2,screen_eccen_deg/2], \
                     [screen_eccen_deg/2, screen_eccen_deg/2],color=boxcolor)
            plt.plot([-screen_eccen_deg/2,screen_eccen_deg/2], \
                     [-screen_eccen_deg/2, -screen_eccen_deg/2],color=boxcolor)

            if pi in pi2label:
                plt.xlabel('x coord (deg)')
                plt.ylabel('y coord (deg)')
            else:
                plt.xticks([])
                plt.yticks([])
            plt.title('%s (%d vox)'%(roi_names[rr], len(inds2use)))

    plt.suptitle('pRF estimates\nshowing all voxels with R2 > %.2f\n%s\n%s'\
                 %(r2_cutoff, summary_plots.get_substr(out), fitting_type));

    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'spatial_prf_distrib.pdf'))
        plt.savefig(os.path.join(fig_save_folder,'spatial_prf_distrib.png'))



def plot_size_vs_eccen(fitting_type, out, roi_def, skip_inds=None, r2_cutoff=0.10, \
                       screen_eccen_deg = 8.4, size_lims=None, \
                       eccen_lims=None, fig_save_folder=None):
    """
    Create a scatter plot for each ROI, showing the size of each voxel's 
    best pRF estimate versus its eccentricity.
    """

    if hasattr(out, 'keys'):        
        # single subject case
        best_ecc_deg,best_ang_deg,best_size_deg = get_prf_pars_deg(out, screen_eccen_deg)
        val_r2 = out['val_r2'][:,0]
    else:        
        # multi subject case       
        best_ecc_deg = [];
        best_angle_deg = [];
        best_size_deg = [];
        for si in range(len(out)):
            be,ba,bs = get_prf_pars_deg(out[si], screen_eccen_deg)
            best_ecc_deg = np.concatenate([best_ecc_deg, be], axis=0)
            best_angle_deg = np.concatenate([best_angle_deg, ba], axis=0)
            best_size_deg = np.concatenate([best_size_deg, bs], axis=0)
        val_r2 = np.concatenate([o['val_r2'][:,0] for o in out])   
   
    n_rois = roi_def.n_rois
    roi_names = roi_def.roi_names
    
    if size_lims is None:
        size_lims = [-1, 1.25*np.max(best_size_deg)]
    if eccen_lims is None:
        eccen_lims = [-1,  1.25*np.max(best_ecc_deg)]
    
    if skip_inds is None:
        skip_inds = []
    
    abv_thresh = val_r2>r2_cutoff
    
    npy = int(np.ceil(np.sqrt(n_rois-len(skip_inds))))
    npx = int(np.ceil((n_rois-len(skip_inds))/npy))
    pi2label = [(npx-1)*npy+1]
    
    plt.figure(figsize=(24,20))

    pi=0
    for rr in range(n_rois):

        if rr not in skip_inds:
            
            inds_this_roi = roi_def.get_indices(rr)
            
            inds2use = np.where(inds_this_roi & abv_thresh)[0]

            pi+=1
            plt.subplot(npx,npy,pi)
            ax = plt.gca()

            xvals = best_ecc_deg[inds2use]
            yvals = best_size_deg[inds2use]
            plt.plot(xvals, yvals, '.')

            # quick linear regression to get a best fit line
            X = np.concatenate([xvals[:,np.newaxis], np.ones((len(inds2use),1))], axis=1)
            y = yvals[:,np.newaxis]
            linefit =  np.linalg.pinv(X) @ y
            
            yhat = xvals*linefit[0] + linefit[1]
            plt.plot(xvals, yhat, '-', color=[0.6, 0.6, 0.6])
            
            plt.xlim(eccen_lims)
            plt.ylim(size_lims)
            
            if pi in pi2label:
                plt.xlabel('eccen (deg)')
                plt.ylabel('size (deg)')
            else:
                plt.xticks([])
                plt.yticks([])

            plt.axhline(0,color=[0.8, 0.8, 0.8])
            plt.axvline(0,color=[0.8, 0.8, 0.8])
            plt.title('%s (%d vox)'%(roi_names[rr], len(inds2use)))

    plt.suptitle('pRF estimates\nshowing all voxels with R2 > %.2f\n%s\n%s'\
                 %(r2_cutoff, summary_plots.get_substr(out), fitting_type))
    
    if fig_save_folder is not None:
        plt.savefig(os.path.join(fig_save_folder,'size_vs_eccen.png'))
        plt.savefig(os.path.join(fig_save_folder,'size_vs_eccen.pdf'))
        
    
def plot_prf_grid(prf_models, xy_circ = [-0.4, -0.4], screen_eccen_deg = 8.4):

    """
    Visualize a grid of pRF positions.
    Makes a subplot for each separate pRF size, with the size drawn as a circle.
    prf_models is [n_prfs x 3], where columns are [x,y,sigma]
    xy_circ is the approximate center of the pRF to draw as a circle 
    (for simplicity, others are just dots)
    """
    
    unique_sizes = np.unique(np.round(prf_models[:,2],4))
    plt.figure(figsize=(18,12));

    for si, size in enumerate(unique_sizes):

        inds = np.where(np.round(prf_models[:,2],4)==size)[0]
        
        prf_models_plot = prf_models[inds,:]
        ind = np.argmin(np.abs(prf_models_plot[:,0] - xy_circ[0]) + np.abs(prf_models_plot[:,1] - xy_circ[1]))
        xy_circ_actual = [prf_models_plot[ind,0], prf_models_plot[ind,1]]

        plt.subplot(3,4,si+1)
        ax = plt.gca()
        plt.plot(prf_models[inds,0]*screen_eccen_deg, prf_models[inds,1]*screen_eccen_deg, '.')
        plt.plot(xy_circ_actual[0]*screen_eccen_deg, xy_circ_actual[1]*screen_eccen_deg, '.',color='k')
        circ = matplotlib.patches.Circle((xy_circ_actual[0]*screen_eccen_deg, \
                                          xy_circ_actual[1]*screen_eccen_deg), \
                                          size*screen_eccen_deg, \
                                          color = [0.2, 0.2, 0.2], fill=False)
        ax.add_artist(circ)
        plt.axis('square')
        # plt.xlim([-1.5*screen_eccen_deg, 1.5*screen_eccen_deg])
        # plt.ylim([-1.5*screen_eccen_deg, 1.5*screen_eccen_deg])
        plt.xlim([-0.8*screen_eccen_deg, 0.8*screen_eccen_deg])
        plt.ylim([-0.8*screen_eccen_deg, 0.8*screen_eccen_deg])
        plt.xticks(np.arange(-8,9,4))
        plt.yticks(np.arange(-8,9,4))

        plt.plot([screen_eccen_deg/2,screen_eccen_deg/2], \
                 [screen_eccen_deg/2, -screen_eccen_deg/2],color=[0.8, 0.8, 0.8])
        plt.plot([-screen_eccen_deg/2,-screen_eccen_deg/2], \
                 [screen_eccen_deg/2, -screen_eccen_deg/2],color=[0.8, 0.8, 0.8])
        plt.plot([-screen_eccen_deg/2,screen_eccen_deg/2], \
                 [screen_eccen_deg/2, screen_eccen_deg/2],color=[0.8, 0.8, 0.8])
        plt.plot([-screen_eccen_deg/2,screen_eccen_deg/2], \
                 [-screen_eccen_deg/2, -screen_eccen_deg/2],color=[0.8, 0.8, 0.8])

        if si>7:
            plt.xlabel('x coord (deg)')
        if np.mod(si,4)==0:
            plt.ylabel('y coord (deg)')

        plt.title('pRF sigma=%.2f deg'%(size*screen_eccen_deg))