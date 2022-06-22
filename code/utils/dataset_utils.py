import torch
from torch.utils.data import Dataset, Subset
import sys, os
import numpy as np
import h5py

from utils import roi_utils, nsd_utils, default_paths

"""
Example usage:
ds = dataset_utils.nsd_dataset(subject=1, roi_names = ['V1','V2'], load_edges=True, \
                     load_images=True, load_voxels=True, grayscale=True, load_each_time=True)
ds_trn, ds_val = dataset_utils.get_trn_val_split(ds)
dl_trn = DataLoader(ds_trn, batch_size=100, shuffle=True, num_workers=0)
"""

class nsd_dataset(Dataset):
  
    """
    Dataset class that loads trialwise voxel activation patterns for NSD data and 
    images for the corresponding trials.
    
    subject:          number of NSD subject, 1-8
    roi_names:        list of ROIs to include (e.g. ['V1', 'FFA-1'])
    load_voxels:      load beta weights for each trial?
    load_images:      load images for each trial?
    load_edges:       load edge map (from sketch tokens method) for the image on each trial?
    load_each_time:   do you want to read items from h5py files each time a new batch is requested, 
                      or load entire file into memory at once when first batch is loaded?
                      can speed up later batch loading, depending on how much memory available.
    grayscale:        convert images to grayscale? if false returns RGB
    image_preproc_fn: specify a custom preprocessing fn. This will over-ride grayscale param.
    
    """ 

    def __init__(self, subject, roi_names=None, load_voxels=True, load_images=True, load_edges=True, \
                  load_each_time=True, grayscale=True, image_preproc_fn=None, dtype=np.float32):
    
        self.subject = subject;
        if (roi_names==[]) or (roi_names==''):
            roi_names = None
        if roi_names is not None and not isinstance(roi_names, list):
            raise ValueError('roi_names should be a list of strings')
        self.roi_names = roi_names;
        self.dtype=dtype
            
        # can choose to return just some of these items (voxel data, full images, edge maps)
        self.load_edges=load_edges
        self.load_images=load_images
        self.load_voxels=load_voxels
        
        self.load_each_time = load_each_time
        if not self.load_each_time:
            # if this is true, then will load from h5py file every time we want a new item, 
            # and not use too much memory. If false, then will load big file into memory 
            # instead of accessing each time. Can be faster but might make memory run out. 
            self.init_big()

        self.trials_per_sess=nsd_utils.trials_per_sess
        self.num_sess=nsd_utils.max_sess_each_subj[self.subject-1]
        
        self.betas_file = os.path.join(default_paths.nsd_data_concat_root, \
                                       'S%d_allsess_concat_visual.h5py'%self.subject)
        if self.load_voxels and not os.path.exists(self.betas_file):
            raise ValueError('looking for beta weights at %s, not found'%self.betas_file)
        self.get_vox_info()
        
        self.stim_file = os.path.join(default_paths.stim_root, 'S%d_stimuli_240.h5py'%self.subject)
        if self.load_images and not os.path.exists(self.stim_file):
            raise ValueError('looking for images at %s, not found'%self.stim_file)
        with h5py.File(self.stim_file,'r') as data_set:
            self.n_pix = data_set['/stimuli'].shape[2]
            data_set.close()
            
        self.edges_file = os.path.join(default_paths.sketch_token_feat_path,'S%d_edges_240.h5py'%self.subject)
        if self.load_edges and not os.path.exists(self.edges_file):
            raise ValueError('looking for edge map images at %s, not found'%self.edges_file)
            
        self.image_order = nsd_utils.get_master_image_order()
        self.image_order = self.image_order[0:len(self)]
        
        if image_preproc_fn is not None:
            # use a custom preproc fn
            self.image_preproc_fn = image_preproc_fn
        else:
            # if not specified, use grayscale param to pick one of these
            if grayscale==True:
                self.image_preproc_fn = nsd_utils.image_uncolorize_fn
            else:
                self.image_preproc_fn = nsd_utils.image_preproc_fn
         
    def get_vox_info(self):

        """
        Determine which voxels to use and how many voxels there are.
        """ 
        
        with h5py.File(self.betas_file, 'r') as data_set:
            ds_size = data_set['/betas'].shape
            data_set.close()
        self.max_vox = ds_size[1]
        assert(self.trials_per_sess * self.num_sess==ds_size[0])

        if self.roi_names is not None:
            # choose voxels in any of the listed rois
            retlabs, facelabs, placelabs, bodylabs, \
            ret_group_names, face_names, place_names, body_names = \
                    roi_utils.get_combined_rois(self.subject, volume_space=True, include_all=True, \
                    include_body=True, verbose=False)
            assert(len(retlabs)==self.max_vox)
            
            voxel_inds_this_roi = np.zeros((len(retlabs),), dtype=bool)
            for rr, rname in enumerate(self.roi_names):
                if rname in ret_group_names:
                    names = ret_group_names;
                    labs = retlabs;          
                elif rname in face_names:
                    names = face_names;
                    labs = facelabs;
                elif rname in place_names:
                    names = place_names;
                    labs = placelabs;
                elif rname in body_names:
                    names = body_names;
                    labs = bodylabs;
                else:
                    raise ValueError('roi_name %s not included in any of name lists',rname)
                ind = [ii for ii in range(len(names)) if names[ii]==rname]
                assert(len(ind)==1)
                ind = ind[0]
                voxel_inds_this_roi = voxel_inds_this_roi | (labs==ind)
                
            self.voxel_mask = voxel_inds_this_roi           
        else:
            # otherwise using all voxels in the big visual ROI
            self.voxel_mask = np.ones((self.max_vox,))==1
        
        self.num_vox = np.sum(self.voxel_mask)
            
    def init_big(self):
        
        # these are big arrays for all trials/all images, from which 
        # individual items can be selected (if load_each_time==False)
        self.vox_big = None
        self.ims_big = None
        self.edges_big = None

    def image_preproc(self, image_batch):
        
        if self.image_preproc_fn is None:
            return image_batch
        else:
            return self.image_preproc_fn(image_batch)
    
    def __len__(self):

        mylen = self.num_sess*self.trials_per_sess
        return mylen

    def __getitem__(self, trial_inds):

        """
        returns a dictionary:
        
        voxel_data:    [n_trials x num_vox] (or None if load_voxels is False)
        image_data:    [n_trials x 1 x n_pix x n_pix] if grayscale=True
                       or [n_trials x 3 x n_pix x n_pix] if grayscale=False
                        (or None if load_images is False)
        edge_map_data: [n_trials x 1 x n_pix x n_pix] (or None if load_edges is False)

        """
        
        if hasattr(trial_inds, '__len__')==True:
            n_trials = len(trial_inds)           
        else:
            n_trials=1
            trial_inds = np.array([trial_inds])
        
        # trial_inds are 0-30000, image_inds are 0-10000 (because each image is seen 3x)
        image_inds = self.image_order[trial_inds]
        
        if self.load_each_time:

            if self.load_voxels:
                values = np.zeros((n_trials,self.max_vox),dtype=self.dtype)                
                with h5py.File(self.betas_file, 'r') as data_set:
                    for ii, trial_ind in enumerate(trial_inds):
                        values[ii,:] = np.array(data_set['/betas'][trial_ind,:]);
                    data_set.close()

                # [batch_size x n_voxels]
                voxel_data = values[:,self.voxel_mask];
            else:
                voxel_data = None
                
            if self.load_images:               
                values = np.zeros((n_trials,3,self.n_pix,self.n_pix),dtype=self.dtype)  
                with h5py.File(self.stim_file, 'r') as data_set:
                    for ii, image_ind in enumerate(image_inds):
                        values[ii,:,:,:] = np.array(data_set['/stimuli'][image_ind,:,:,:]);
                    data_set.close()
                # [batch_size x 1 x h x w]
                image_data = self.image_preproc(values)
            else:
                image_data = None
                
            if self.load_edges:
                values = np.zeros((self.n_pix,self.n_pix,n_trials),dtype=self.dtype)  
                with h5py.File(self.edges_file, 'r') as data_set:
                    for ii, image_ind in enumerate(image_inds):
                        values[:,:,ii] = np.array(data_set['/features'][:,:,image_ind]);
                    data_set.close()
                # [batch_size x 1 x h x w]
                edge_map_data = np.expand_dims(np.moveaxis(values, [0,1,2],[1,2,0]),axis=1)
            else:
                edge_map_data = None
            
        else:
            
            if self.load_voxels:                
                if self.vox_big is None:
                    print('loading voxel data')
                    with h5py.File(self.betas_file, 'r') as data_set:
                        values = np.array(data_set['/betas'][:,:]);
                        data_set.close()                    
                    self.vox_big = values[:,self.voxel_mask]
                # [batch_size x n_voxels]
                voxel_data = self.vox_big[trial_inds,:]
            else:
                voxel_data = None
               
            if self.load_images:                
                if self.ims_big is None:
                    print('loading image data')
                    with h5py.File(self.stim_file, 'r') as data_set:
                        values = np.array(data_set['/stimuli'][:,:,:,:]);
                        data_set.close()                  
                    self.ims_big = self.image_preproc(values)
                # [batch_size x 1 x h x w]
                image_data = self.ims_big[image_inds,:,:,:]
            else:
                image_data = None
                
            if self.load_edges:                
                if self.edges_big is None:
                    print('loading edge maps')
                    with h5py.File(self.edges_file, 'r') as data_set:
                        values = np.array(data_set['/features'][:,:,:]);
                        data_set.close()                  
                    self.edges_big = np.expand_dims(np.moveaxis(values, [0,1,2],[1,2,0]),axis=1)
                # [batch_size x 1 x h x w]
                edge_map_data = self.edges_big[image_inds,:,:,:]
            else:
                edge_map_data = None
                
        if n_trials==1:
            # remove the first singleton dimension here, otherwise batches returned by 
            # dataloader will have extra dim.
            voxel_data = np.squeeze(voxel_data, axis=0)
            image_data = np.squeeze(image_data, axis=0)
            edge_map_data = np.squeeze(edge_map_data, axis=0)
            
        item = {'voxel_data': voxel_data, 'image_data': image_data, 'edge_map_data': edge_map_data}
        
        return item 

    
def get_trn_val_split(dataset):
    
    """
    For an NSD dataset, separate into train/val trials.
    The first 1000 images are shared across all subjects - these are always held out as
    validation data. Note some subjects are missing a few sessions, so their 
    train/validation sets are both smaller.
    """
    
    val_inds = np.where(dataset.image_order<1000)[0]
    trn_inds = np.where(dataset.image_order>=1000)[0]
    ds_val = Subset(dataset, val_inds)
    ds_trn = Subset(dataset, trn_inds)
    
    return ds_trn, ds_val