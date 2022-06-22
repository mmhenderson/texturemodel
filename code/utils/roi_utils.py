import numpy as np
import os, sys
import pandas as pd
import nibabel as nib
import copy

ret_group_names = ['V1', 'V2', 'V3','hV4','VO1-2','PHC1-2','LO1-2','TO1-2','V3ab',\
                   'IPS0-1','IPS2-5','SPL1','FEF']
ret_group_inds = [[1,2],[3,4],[5,6],[7],[8,9],[10,11],[14,15],[12,13],[16,17],[18,19],[20,21,22,23],[24],[25]]

from utils import default_paths
from utils import nsd_utils

class nsd_roi_def():
    
    """
    An object containing ROI definitions for a subject in NSD dataset.
    The ROIs included here are retinotopic visual regions (defined using a combination of 
    Kastner 2015 atlas and pRF mapping data), and category-selective (face, place, body) ROIs.
    Definitions will be partly overlapping between different methods 
    (for instance a voxel can be in both V3A and OPA)
    If skip_areas is provided, this is an index into the original n_rois list, for which
    you would like to skip here. Will reduce the total n_rois.
    """
    
    def __init__(self, subject, volume_space=True, use_kastner_areas = True, \
                 areas_include=None, areas_merge = None, \
                 remove_ret_overlap = False, remove_categ_overlap=False):
        
        self.subject=subject
        self.volume_space=volume_space
        self.use_kastner_areas=use_kastner_areas
        
        self.__init_names__()        
        self.__init_labels__()

        if areas_include is None:            
            # convenience option, these are the areas we generally want to use
            if self.use_kastner_areas:
                self.areas_include = ['V1','V2','V3','hV4',\
                                      'V3ab','IPS',\
                                      'OPA','PPA','RSC',\
                                      'OFA','FFA','EBA']
                
            else:
                self.areas_include = ['V1','V2','V3','hV4', \
                                      'OPA','PPA','RSC', \
                                      'OFA','FFA','EBA']
        else:
            assert(isinstance(areas_include, list) or isinstance(areas_include, np.ndarray))
            assert(isinstance(areas_include[0], str))
            self.areas_include = areas_include
           
        if areas_merge is None:
            if self.use_kastner_areas:          
                self.areas_merge = [ [['IPS0-1','IPS2-5'], ['FFA-1','FFA-2']], \
                                ['IPS',                'FFA']]
            else:
                self.areas_merge = [ [['FFA-1','FFA-2']], ['FFA']]
        else:
            self.areas_merge = areas_merge
            
    
        for a1, a2 in zip(self.areas_merge[0], self.areas_merge[1]):
            # combining these sub-regions to make larger areas, for simplicity
            self.merge_two_areas(a1[0], a1[1], a2)
            
        if np.any([area not in self.roi_names for area in self.areas_include]):
            print('at least one roi name in areas_include not recognized, inputs were: ')
            print(self.areas_include)
            return
        
        skip_areas = [ri for ri in range(len(self.roi_names)) \
                          if self.roi_names[ri] not in self.areas_include]    
        if len(skip_areas)>0:
            # ignore any areas we didn't specify
            self.__remove_skipped__(skip_areas)
  
        if remove_ret_overlap:
            self.__remove_ret_overlap__()
        if remove_categ_overlap:
            self.__remove_categ_overlap__()
            
    def __init_names__(self):
        
        ret, face, place, body = load_roi_label_mapping(self.subject)
        self.face_names = face[1]
        self.place_names = place[1]
        self.body_names = body[1]
        if self.use_kastner_areas:
            # including approximate definitions for many areas based on atlas
            self.ret_names = copy.deepcopy(ret_group_names)
        else:
            # including pRF-based definitions, only through V4
            self.ret_names = [copy.deepcopy(ret_group_names[ii]) for ii in range(4)]
            
        self.__combine_names__()
       
    def __combine_names__(self):
        
        self.nret = len(self.ret_names)
        self.nplace = len(self.place_names)
        self.nface = len(self.face_names)        
        self.nbody = len(self.body_names)
        
        self.n_rois = len(self.ret_names) + len(self.face_names) \
                        + len(self.place_names) + len(self.body_names)
        self.roi_names = self.ret_names+self.place_names+self.face_names+self.body_names

        self.is_ret = np.arange(0, self.n_rois)<self.nret
        self.is_place = (np.arange(0, self.n_rois)>=self.nret) & \
                        (np.arange(0, self.n_rois)<self.nret+self.nplace)
        self.is_face = (np.arange(0, self.n_rois)>=self.nret+self.nplace) & \
                        (np.arange(0, self.n_rois)<self.nret+self.nplace+self.nface)        
        self.is_body = np.arange(0, self.n_rois)>=self.nret+self.nface+self.nplace
        
    def __init_labels__(self):
        
        voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = \
                    get_voxel_roi_info(self.subject, \
                                       volume_space=self.volume_space, \
                                       use_kastner_areas=self.use_kastner_areas)
        self.voxel_mask = voxel_mask
        self.nii_shape = brain_nii_shape
        
        [roi_labels_retino, roi_labels_face, roi_labels_place, roi_labels_body] = \
                    copy.deepcopy(voxel_roi)
        
        # make these zero-indexed, where 0 is first ROI and -1 is not in any ROI
        self.placelabs = roi_labels_place[voxel_index] - 1
        self.placelabs[self.placelabs==-2] = -1
        self.facelabs = roi_labels_face[voxel_index] - 1        
        self.facelabs[self.facelabs==-2] = -1        
        self.bodylabs = roi_labels_body[voxel_index] - 1
        self.bodylabs[self.bodylabs==-2] = -1

        roi_labels_retino = roi_labels_retino[voxel_index]
        self.retlabs = (-1)*np.ones(np.shape(roi_labels_retino))

        for rr in range(len(ret_group_names)):   
            inds_this_roi = np.isin(roi_labels_retino, ret_group_inds[rr])
            self.retlabs[inds_this_roi] = rr

    def __remove_skipped__(self, skip_areas):
        
        rc=0; fc=0; pc=0; bc=0;
        for aa in skip_areas:
            if self.is_ret[aa]:
                ind = aa-rc
                self.ret_names.remove(self.ret_names[ind])
                self.retlabs[self.retlabs==ind] = -1
                self.retlabs[self.retlabs>ind] -= 1
                rc+=1
            elif self.is_place[aa]:
                ind = aa-self.nret-pc
                self.place_names.remove(self.place_names[ind])
                self.placelabs[self.placelabs==ind] = -1
                self.placelabs[self.placelabs>ind] -= 1
                pc+=1
            elif self.is_face[aa]:
                ind = aa-self.nret-self.nplace-fc
                self.face_names.remove(self.face_names[ind])
                self.facelabs[self.facelabs==ind] = -1
                self.facelabs[self.facelabs>ind] -= 1
                fc+=1           
            elif self.is_body[aa]:
                ind = aa-self.nret-self.nface-self.nplace-bc
                self.body_names.remove(self.body_names[ind])
                self.bodylabs[self.bodylabs==ind] = -1
                self.bodylabs[self.bodylabs>ind] -= 1
                bc+=1
            
        self.__combine_names__()
        
    def __remove_ret_overlap__(self):
    
        self.retlabs[self.facelabs>-1] = -1
        self.retlabs[self.placelabs>-1] = -1
        self.retlabs[self.bodylabs>-1] = -1
        
    def __remove_categ_overlap__(self):

        self.placelabs[self.facelabs>-1] = -1
        self.bodylabs[self.facelabs>-1] = -1
        self.bodylabs[self.placelabs>-1] = -1
       
    def merge_two_areas(self, roi_name1, roi_name2, roi_name_combined):
        
        if np.any([name==roi_name1 for name in self.roi_names]):
            rr1 = np.where([name==roi_name1 for name in self.roi_names])[0][0]
        else:
            raise ValueError('%s not in list of ROIs, see self.roi_names for options.'%roi_name1)
        
        if np.any([name==roi_name2 for name in self.roi_names]):
            rr2 = np.where([name==roi_name2 for name in self.roi_names])[0][0]
        else:
            raise ValueError('%s not in list of ROIs, see self.roi_names for options.'%roi_name2)
            
        if self.is_ret[rr1]:
            assert(self.is_ret[rr2])
            ind1 = rr1
            ind2 = rr2
            self.ret_names[ind1] = roi_name_combined
            self.ret_names.remove(self.ret_names[ind2])
            self.retlabs[self.retlabs==ind2] = ind1
            self.retlabs[self.retlabs>ind2] -= 1
        elif self.is_place[rr1]:
            assert(self.is_place[rr2])
            ind1 = rr1-self.nret
            ind2 = rr2-self.nret
            self.place_names[ind1] = roi_name_combined
            self.place_names.remove(self.place_names[ind2])
            self.placelabs[self.placelabs==ind2] = ind1
            self.placelabs[self.placelabs>ind2] -= 1
        elif self.is_face[rr1]:
            assert(self.is_face[rr2])
            ind1 = rr1-self.nret-self.nplace
            ind2 = rr2-self.nret-self.nplace
            self.face_names[ind1] = roi_name_combined
            self.face_names.remove(self.face_names[ind2])
            self.facelabs[self.facelabs==ind2] = ind1
            self.facelabs[self.facelabs>ind2] -= 1
        else:
            assert(self.is_body[rr1] and self.is_body[rr2])
            ind1 = rr1-self.nret-self.nplace-self.nface
            ind2 = rr2-self.nret-self.nplace-self.nface
            self.body_names[ind1] = roi_name_combined
            self.body_names.remove(self.body_names[ind2])
            self.bodylabs[self.bodylabs==ind2] = ind1
            self.bodylabs[self.bodylabs>ind2] -= 1
        
        self.__combine_names__()
        
    def get_indices_from_name(self, roi_name):
        
        if np.any([name==roi_name for name in self.roi_names]):
            rr = np.where([name==roi_name for name in self.roi_names])[0][0]
            return self.get_indices(rr)
        else:
            raise ValueError('%s not in list of ROIs, see self.roi_names for options.'%roi_name)
        
    def get_indices(self, rr):
        
        # rr is an index into self.roi_names, in range 0-self.n_rois
        if (rr<0) or (rr>(self.n_rois-1)):
            raise ValueError('rr needs to be between 0-%d'%self.n_rois)
        
        if self.is_ret[rr]:
            inds_this_roi = self.retlabs==rr
        elif self.is_place[rr]:
            inds_this_roi = self.placelabs==(rr-self.nret)
        elif self.is_face[rr]:
            inds_this_roi = self.facelabs==(rr-self.nret-self.nplace)        
        elif self.is_body[rr]:
            inds_this_roi = self.bodylabs==(rr-self.nret-self.nface-self.nplace)

        return inds_this_roi
    
    def get_sizes(self):

        n_each = np.zeros((self.n_rois,),dtype=int)
        for rr in range(self.n_rois):
            inds_this_roi = self.get_indices(rr)           
            n_total = np.sum(inds_this_roi)
            n_each[rr] = n_total

        return n_each

    def print_overlap(self):

        for rr in range(self.n_rois):
            inds_this_roi = self.get_indices(rr)         
            n_total = np.sum(inds_this_roi)
            print('%s: %d vox total'%(self.roi_names[rr], n_total))
            for rr2 in range(self.n_rois):                
                if rr2==rr:
                    continue                   
                inds_this_roi2 = self.get_indices(rr2)
                n_overlap = np.sum(inds_this_roi & inds_this_roi2)
                if n_overlap>0:
                    print('    %d vox overlap with %s'%(n_overlap, self.roi_names[rr2]))

                    
class multi_subject_roi_def(nsd_roi_def):
    
    """
    A class for combining ROI definitions across multiple subjects.
    To be used for cases where we have concatenated some property (i.e. encoding model fit
    performance) across all voxels in all subjects. 
    Can use "get_indices" same way as for single subject case - here it will return a long
    boolean array with length equal to the total number of voxels across all subjects.
    Just make sure that the order of subjects here (arg 'subjects') is same as it was when 
    concatenating the property of interest for analysis.
    """
    
    def __init__(self, subjects, volume_space=True, use_kastner_areas = True, \
                 areas_include=None, areas_merge=None, \
                 remove_ret_overlap = False, remove_categ_overlap=False):
     
        # first initialize object with just first subject, most of
        # the properties are same for all subs (ROI names etc.)
        super().__init__(subject=subjects[0], \
                         volume_space=volume_space, \
                         use_kastner_areas = use_kastner_areas, \
                         areas_include=areas_include, \
                         areas_merge=areas_merge, \
                         remove_ret_overlap = remove_ret_overlap, \
                         remove_categ_overlap=remove_categ_overlap)
        
        self.subjects = subjects
        
        # now getting subject-specific properties, labels for each voxel.
        self.ss_roi_defs = [nsd_roi_def(subject=ss, \
                                        volume_space=volume_space, \
                                        use_kastner_areas = use_kastner_areas, \
                                        areas_include=areas_include, \
                                        areas_merge=areas_merge, \
                                        remove_ret_overlap = remove_ret_overlap, \
                                        remove_categ_overlap=remove_categ_overlap) \
                            for ss in self.subjects]
        
        self.__concat_labels__()
        
        self.merge_two_areas = self.__merge_two_areas__
        
    def __concat_labels__(self):

        # concatenate ROI labels across all subjects.
        self.facelabs = np.concatenate([roi.facelabs for roi in self.ss_roi_defs], axis=0)
        self.placelabs = np.concatenate([roi.placelabs for roi in self.ss_roi_defs], axis=0)
        self.bodylabs = np.concatenate([roi.bodylabs for roi in self.ss_roi_defs], axis=0)
        self.retlabs = np.concatenate([roi.retlabs for roi in self.ss_roi_defs], axis=0)
        
        # these properties will be in lists over subjects
        self.voxel_mask = [roi.voxel_mask for roi in self.ss_roi_defs]
        self.nii_shape = [roi.nii_shape for roi in self.ss_roi_defs]
        
        self.roi_names = self.ss_roi_defs[0].roi_names
        self.place_names = self.ss_roi_defs[0].place_names
        self.face_names = self.ss_roi_defs[0].face_names        
        self.body_names = self.ss_roi_defs[0].body_names
        
    def __merge_two_areas__(self, roi_name1, roi_name2, roi_name_combined):
        
        for roi_def in self.ss_roi_defs:
            roi_def.merge_two_areas(roi_name1, roi_name2, roi_name_combined)
            
        self.__concat_labels__()
        self.__combine_names__()
        
        
        
def load_roi_label_mapping(subject):
    """
    Load files (ctab) that describe the mapping from numerical labels to text labels.
    These correspond to the mask definitions of each type of ROI (either nii or mgz files).
    This code will get mappings for pRF ROIs, Kastner atlas ROIs, floc-faces, 
    floc-places, and floc-bodies.
    """
    
    filename_prf = os.path.join(default_paths.nsd_root,'nsddata','freesurfer',\
                                'subj%02d'%subject, 'label', 'prf-visualrois.mgz.ctab')
    names = np.array(pd.read_csv(filename_prf))
    names = [str(name) for name in names]
    prf_num_labels = [int(name[2:np.char.find(name,' ')]) for name in names]
    prf_text_labels=[]
    for name in names:
        if np.char.find(name,'\\')>-1:
            prf_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,'\\')])
        else:
            prf_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,"'",2)])
   
    filename_ret = os.path.join(default_paths.nsd_root,'nsddata','freesurfer',\
                                'subj%02d'%subject, 'label', 'Kastner2015.mgz.ctab')
    names = np.array(pd.read_csv(filename_ret))
    names = [str(name) for name in names]
    ret_num_labels = [int(name[2:np.char.find(name,' ')]) for name in names]
    ret_text_labels=[]
    for name in names:
        if np.char.find(name,'\\')>-1:
            ret_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,'\\')])
        else:
            ret_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,"'",2)])
    
    # kastner atlas and prf have same values/names for all shared elements, 
    # so can just use kastner going forward.
    assert(np.array_equal(prf_num_labels,ret_num_labels[0:len(prf_num_labels)]))
    assert(np.array_equal(prf_text_labels,ret_text_labels[0:len(prf_text_labels)]))

    filename_faces = os.path.join(default_paths.nsd_root,'nsddata','freesurfer',\
                                  'subj%02d'%subject, 'label', 'floc-faces.mgz.ctab')
    names = np.array(pd.read_csv(filename_faces))
    names = [str(name) for name in names]
    faces_num_labels = [int(name[2:np.char.find(name,' ')]) for name in names]
    faces_text_labels=[]
    for name in names:
        if np.char.find(name,'\\')>-1:
            faces_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,'\\')])
        else:
            faces_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,"'",2)])
    
    filename_places = os.path.join(default_paths.nsd_root,'nsddata','freesurfer',\
                                   'subj%02d'%subject, 'label', 'floc-places.mgz.ctab')
    names = np.array(pd.read_csv(filename_places))
    names = [str(name) for name in names]
    places_num_labels = [int(name[2:np.char.find(name,' ')]) for name in names]
    places_text_labels=[]
    for name in names:
        if np.char.find(name,'\\')>-1:
            places_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,'\\')])
        else:
            places_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,"'",2)])
       
    filename_body = os.path.join(default_paths.nsd_root,'nsddata','freesurfer',\
                                 'subj%02d'%subject, 'label', 'floc-bodies.mgz.ctab')
    names = np.array(pd.read_csv(filename_body))
    names = [str(name) for name in names]
    body_num_labels = [int(name[2:np.char.find(name,' ')]) for name in names]
    body_text_labels=[]
    for name in names:
        if np.char.find(name,'\\')>-1:
            body_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,'\\')])
        else:
            body_text_labels.append(name[np.char.find(name,' ')+1:np.char.find(name,"'",2)])
    
    return [ret_num_labels, ret_text_labels], [faces_num_labels, faces_text_labels], \
            [places_num_labels, places_text_labels], [body_num_labels, body_text_labels]
        
def get_nii_shape(subject):
    """
    Get the original shape of 3D nifti files for NSD data, for a given subject.
    """
    
    # can load any nifti file for this subject.
    roi_path = os.path.join(default_paths.nsd_root, 'nsddata', 'ppdata', \
                                'subj%02d'%subject, 'func1pt8mm', 'roi')
    prf_labels_full  = nsd_utils.load_from_nii(os.path.join(roi_path, 'prf-visualrois.nii.gz'))
    # save the shape, so we can project back to volume space later.
    brain_nii_shape = np.array(prf_labels_full.shape)
    
    return brain_nii_shape

def get_voxel_roi_info(subject, volume_space=True, use_kastner_areas=True):

    """
    For a specified NSD subject, load all definitions of all ROIs.
    The ROIs included here are retinotopic visual regions (defined using a combination of 
    Kastner 2015 atlas and pRF mapping data), and category-selective (face, body, place) ROIs.
    Will return four separate bricks of labels, for each method of definition (retino, category, etc.)
    These are partially overlapping, so can choose later which definition to use for the overlapping voxels.
    Can be done in either volume space (volume_space=True) or surface space (volume_space=False).
    If surface space, then each voxel is a "vertex" of mesh.
    
    Inputs:
        subject (int): NSD subject number, 1-8
        volume_space (bool, default=True): working with 3D volume space data (else mesh vertices)
        
    Outputs:
        voxel_mask (1D boolean array): indicates which of the voxels to use for analysis. These 
            are the voxels identified as belonging to any ROI or to the NSDgeneral mask.
        voxel_index (1D array of ints): indices included in the mask (i.e. np.where(voxel_mask))
        [roi_labels_retino, roi_labels_face, roi_labels_place, roi_labels_body]:
            Each is a 1D array of ints, same size as voxel_index.
            Indicates which ROI each voxel belongs to, within the relevant naming scheme. 
            See load_roi_label_mapping() for what the numbers mean.
        nscnr_full (1D array): estimate of NCSNR for each voxel in whole brain
        nii_shape (3-tuple): original size of the nifti files, before flattening. 
            (if volume_space=False, this is None)
    
    """
    
     # First loading each ROI definitions file - lists nvoxels long, with diff numbers for each ROI.
    if volume_space:

        roi_path = os.path.join(default_paths.nsd_root, 'nsddata', 'ppdata', \
                                'subj%02d'%subject, 'func1pt8mm', 'roi')
       
        nsd_general_full = nsd_utils.load_from_nii(os.path.join(roi_path, 'nsdgeneral.nii.gz')).flatten()
            
        prf_labels_full  = nsd_utils.load_from_nii(os.path.join(roi_path, 'prf-visualrois.nii.gz'))
        # save the shape, so we can project back to volume space later.
        brain_nii_shape = np.array(prf_labels_full.shape)
        prf_labels_full = prf_labels_full.flatten()

        kast_labels_full = nsd_utils.load_from_nii(os.path.join(roi_path, 'Kastner2015.nii.gz')).flatten()
        face_labels_full = nsd_utils.load_from_nii(os.path.join(roi_path, 'floc-faces.nii.gz')).flatten()
        place_labels_full = nsd_utils.load_from_nii(os.path.join(roi_path, 'floc-places.nii.gz')).flatten()
        body_labels_full = nsd_utils.load_from_nii(os.path.join(roi_path, 'floc-bodies.nii.gz')).flatten()
        
        # Masks of ncsnr values for each voxel 
        ncsnr_full = nsd_utils.load_from_nii(os.path.join(default_paths.beta_root, \
                                              'subj%02d'%subject, 'func1pt8mm', \
                                              'betas_fithrf_GLMdenoise_RR', 'ncsnr.nii.gz')).flatten()
    else:
        
        roi_path = os.path.join(default_paths.nsd_root,'nsddata', 'freesurfer', 'subj%02d'%subject, 'label')

        # Surface space, concatenate the two hemispheres
        # always go left then right, to match the data which also gets concatenated same way
        prf_labs1 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'lh.prf-visualrois.mgz'))[:,0,0]
        prf_labs2 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'rh.prf-visualrois.mgz'))[:,0,0]
        prf_labels_full = np.concatenate((prf_labs1, prf_labs2), axis=0)

        kast_labs1 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'lh.Kastner2015.mgz'))[:,0,0]
        kast_labs2 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'rh.Kastner2015.mgz'))[:,0,0]
        kast_labels_full = np.concatenate((kast_labs1, kast_labs2), axis=0)

        face_labs1 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'lh.floc-faces.mgz'))[:,0,0]
        face_labs2 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'rh.floc-faces.mgz'))[:,0,0]
        face_labels_full = np.concatenate((face_labs1, face_labs2), axis=0)

        place_labs1 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'lh.floc-places.mgz'))[:,0,0]
        place_labs2 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'rh.floc-places.mgz'))[:,0,0]
        place_labels_full = np.concatenate((place_labs1, place_labs2), axis=0)
      
        body_labs1 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'lh.floc-bodies.mgz'))[:,0,0]
        body_labs2 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'rh.floc-bodies.mgz'))[:,0,0]
        body_labels_full = np.concatenate((body_labs1, body_labs2), axis=0)
      
        # Note this part hasn't been tested
        general_labs1 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'lh.nsdgeneral.mgz'))[:,0,0]
        general_labs2 = nsd_utils.load_from_mgz(os.path.join(roi_path, 'rh.nsdgeneral.mgz'))[:,0,0]
        nsd_general_full = np.concatenate((general_labs1, general_labs2), axis=0)
  
        # Masks of ncsnr values for each voxel 
        n1 = nsd_utils.load_from_mgz(os.path.join(default_paths.beta_root, \
                                                  'subj%02d'%subject, 'nativesurface',\
                                                  'betas_fithrf_GLMdenoise_RR', 'lh.ncsnr.mgh')).flatten()
        n2 = nsd_utils.load_from_mgz(os.path.join(default_paths.beta_root, 'subj%02d'%subject, \
                                                  'nativesurface','betas_fithrf_GLMdenoise_RR',\
                                                  'rh.ncsnr.mgh')).flatten()
        ncsnr_full = np.concatenate((n1, n2), axis=0)
  
        brain_nii_shape = None

    # boolean masks of which voxels had definitions in each of these naming schemes
    has_general_label = (nsd_general_full>0).astype(bool)
    has_prf_label = (prf_labels_full>0).astype(bool)
    has_kast_label = (kast_labels_full>0).astype(bool)
    has_face_label = (face_labels_full>0).astype(bool)
    has_place_label = (place_labels_full>0).astype(bool)
    has_body_label = (body_labels_full>0).astype(bool)
    
    # this is the mask of all the voxels that we want to use for analysis.
    # including any voxels that have ROI defs, OR are in the nsdgeneral mask.
    voxel_mask = has_general_label | has_prf_label | has_kast_label | \
                    has_face_label | has_place_label | has_body_label
    voxel_idx = np.where(voxel_mask) # numerical indices into the big array
    
    # Make our definitions of retinotopic ROIs
    if use_kastner_areas:
        # starting with the Kastner atlas.
        roi_labels_retino = np.copy(kast_labels_full)    
        # Partially overwrite these defs with pRF mapping defs, which are more 
        # accurate when they exist. The numbers have same meaning across these 
        # sets, so this is ok. 
        roi_labels_retino[has_prf_label] = prf_labels_full[has_prf_label]
    else:
        roi_labels_retino = prf_labels_full;
        
    roi_labels_face = face_labels_full
    roi_labels_place = place_labels_full
    roi_labels_body = body_labels_full
 
    return voxel_mask, voxel_idx, [roi_labels_retino, roi_labels_face, roi_labels_place, roi_labels_body], \
                ncsnr_full, brain_nii_shape


                
                  
def view_data(vol_shape, idx_mask, data_vol, order='C', save_to=None):
    view_vol = np.ones(np.prod(vol_shape), dtype=np.float32) * np.nan
    view_vol[idx_mask.astype('int').flatten()] = data_vol
    view_vol = view_vol.reshape(vol_shape, order=order)
    if save_to:
        nib.save(nib.Nifti1Image(view_vol, affine=np.eye(4)), save_to)
    return view_vol
