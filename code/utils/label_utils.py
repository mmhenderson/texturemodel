# import basic modules
import sys
import os
import numpy as np
import pandas as pd
import PIL

from utils import default_paths, nsd_utils, prf_utils, segmentation_utils, coco_utils
from model_fitting import initialize_fitting

def write_binary_labels_csv(subject, stuff=False):
    """
    Creating a csv file where columns are binary labels for the presence/absence of categories
    and supercategories in COCO.
    10,000 images long (same size as image arrays at /user_data/mmhender/nsd_stimuli/stimuli/)
    """
    if subject==999:
        # 999 is a code i am using to indicate the independent set of coco images, which were
        # not actually shown to any NSD participants
        subject_df = coco_utils.load_indep_coco_info()      
    else:
        subject_df = nsd_utils.get_subj_df(subject);
        
    all_coco_ids = np.array(subject_df['cocoId'])

    if stuff:
        coco_object = coco_utils.coco_stuff_val
    else:
        coco_object = coco_utils.coco_val
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = coco_utils.get_coco_cat_info(coco_object)
       
    ims_each_cat, cats_each_image = coco_utils.list_cats_each_image(all_coco_ids, stuff=stuff)
    ims_each_supcat, supcats_each_image = coco_utils.list_supcats_each_image(all_coco_ids, stuff=stuff)

    binary_df = pd.DataFrame(data=np.concatenate([ims_each_supcat, ims_each_cat], axis=1), \
                                 columns = supcat_names + cat_names)

    if not stuff:        
        animate_supcat_ids = [1,9]
        has_animate = np.array([np.any(np.isin(supcats_each_image[ii], animate_supcat_ids)) \
                                for ii in range(len(supcats_each_image))])
        binary_df['has_animate'] = has_animate.astype('int')

        fn2save = os.path.join(default_paths.stim_labels_root, 'S%d_cocolabs_binary.csv'%subject)
    else:
        fn2save = os.path.join(default_paths.stim_labels_root, 'S%d_cocolabs_stuff_binary.csv'%subject)
        
    print('Saving to %s'%fn2save)
    binary_df.to_csv(fn2save, header=True)
    
    return

def write_binary_labels_csv_within_prf(subject, min_pix = 10, stuff=False, \
                                       which_prf_grid=1, debug=False):
    """
    Creating a csv file where columns are binary labels for the presence/absence of categories
    and supercategories in COCO.
    Analyzing presence of categories for each pRF position separately - make a separate csv file for each prf.
    10,000 images long (same size as image arrays in /nsd/stimuli/)
    """

    if subject==999:
        # 999 is a code i am using to indicate the independent set of coco images, which were
        # not actually shown to any NSD participants
        subject_df = coco_utils.load_indep_coco_info()      
    else:
        subject_df = nsd_utils.get_subj_df(subject);
 
    # Params for the spatial aspect of the model (possible pRFs)
    models = initialize_fitting.get_prf_models(which_grid=which_prf_grid)    

    # Get masks for every pRF (circular), in coords of NSD images
    n_prfs = len(models)
    n_pix = 425
    n_prf_sd_out = 2
    prf_masks = np.zeros((n_prfs, n_pix, n_pix))
    
    for prf_ind in range(n_prfs):    
        prf_params = models[prf_ind,:] 
        x,y,sigma = prf_params
        aperture=1.0
        prf_mask = prf_utils.get_prf_mask(center=[x,y], sd=sigma, \
                               patch_size=n_pix)
        prf_masks[prf_ind,:,:] = prf_mask.astype('int')
    
    # the number of pixels required to overlap will depend on how many
    # pixels the pRF occupies.
    mask_sums = np.sum(np.sum(prf_masks, axis=1), axis=1)
#     min_pix_req = np.ceil(mask_sums*min_overlap_pct)
    min_pix_req = min_pix*np.ones((n_prfs,))
    
    # Initialize arrays to store all labels for each pRF
    if stuff:
        coco_v = coco_utils.coco_stuff_val
        coco_t = coco_utils.coco_stuff_trn
    else:
        coco_v = coco_utils.coco_val
        coco_t = coco_utils.coco_trn
        
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = coco_utils.get_coco_cat_info(coco_v)

    n_images = len(subject_df)
    n_categ = len(cat_names)
    n_supcateg = len(supcat_names)
    n_prfs = len(models)
    cat_labels_binary = np.zeros((n_images, n_categ, n_prfs))
    supcat_labels_binary = np.zeros((n_images, n_supcateg, n_prfs))

    for image_ind in range(n_images):

        if debug and image_ind>1:
            continue
            
        print('Processing image %d of %d'%(image_ind, n_images))

        # figure out if it's training or val set and use the right coco api dataset
        if np.array(subject_df['cocoSplit'])[image_ind]=='val2017':
            coco = coco_v
            coco_image_dir = '/lab_data/tarrlab/common/datasets/COCO/val2017'
        else:
            coco = coco_t
            coco_image_dir = '/lab_data/tarrlab/common/datasets/COCO/train2017'

        # for this image, figure out where all the annotations are   
        cocoid = np.array(subject_df['cocoId'])[image_ind]
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[cocoid]))
        masks = np.array([coco.annToMask(annotations[aa]) for aa in range(len(annotations))])

        # get image metadata, for this coco id
        img_info = coco.loadImgs(ids=[cocoid])[0]
        # how was the image cropped to get from coco original to NSD?
        crop_box_pixels = segmentation_utils.get_crop_box_pixels(np.array(subject_df['cropBox'])[image_ind], \
                                                                 [img_info['height'], img_info['width']])

        for aa in range(len(annotations)):

            # Adjust this annotation to match the image's size in NSD (crop/resize)
            mask = masks[aa,:,:]
            mask_cropped = mask[crop_box_pixels[0]:crop_box_pixels[1], crop_box_pixels[2]:crop_box_pixels[3]]
            newsize=[n_pix, n_pix]
            mask_cropped_resized = np.asarray(PIL.Image.fromarray(mask_cropped).resize(newsize,\
                                            resample=PIL.Image.BILINEAR))

            # find where this overlaps with any pRFs
            overlap_pix = np.tensordot(mask_cropped_resized, prf_masks, [[0,1], [1,2]])
            has_overlap = overlap_pix > min_pix_req
            
            if np.any(has_overlap):
                
                cid = annotations[aa]['category_id']
                column_ind = np.where(np.array(cat_ids)==cid)[0][0]
                supcat_column_ind = np.where([np.any(np.isin(ids_each_supcat[sc],cid)) \
                                  for sc in range(n_supcateg)])[0][0]
                
                # Loop over pRFs that overlap
                for prf_ind in np.where(has_overlap)[0]:
                    
                    cat_labels_binary[image_ind, column_ind, prf_ind] = 1                    
                    supcat_labels_binary[image_ind, supcat_column_ind, prf_ind] = 1

        sys.stdout.flush()            
             
                    
    # Now save as csv files for each pRF
    for mm in range(n_prfs):
        
        if debug and mm>1:
            continue

        binary_df = pd.DataFrame(data=np.concatenate([supcat_labels_binary[:,:,mm], \
                                                      cat_labels_binary[:,:,mm]], axis=1), \
                                                      columns = supcat_names + cat_names)
    
        folder2save = os.path.join(default_paths.stim_labels_root, \
                                           'S%d_within_prf_grid%d'%(subject, which_prf_grid))
        if not os.path.exists(folder2save):
            os.makedirs(folder2save)
        if stuff:
            fn2save =  os.path.join(folder2save,'S%d_cocolabs_stuff_binary_prf%d.csv'%(subject, mm))
        else:
            fn2save =  os.path.join(folder2save,'S%d_cocolabs_binary_prf%d.csv'%(subject, mm))
        print('Saving to %s'%fn2save)
        binary_df.to_csv(fn2save, header=True)
        
        
def write_indoor_outdoor_csv(subject):
    """
    Creating binary labels for indoor/outdoor status of images (inferred based on presence of 
    various categories in coco and coco-stuff).
    """
    
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = \
            coco_utils.get_coco_cat_info(coco_utils.coco_val)

    stuff_cat_objects, stuff_cat_names, stuff_cat_ids, stuff_supcat_names, stuff_ids_each_supcat = \
            coco_utils.get_coco_cat_info(coco_utils.coco_stuff_val)

    must_be_indoor = np.array([0, 0, 0, 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, 0, 0, 0, \
                 0, 1, 1, 1, 1, 1, 1, 0, \
                 1, 1, 1, 0, 1, 1, 1, 1, \
                 1, 0, 1, 1, 0, 0, 1, 1 ])

    must_be_outdoor = np.array([0, 0, 1, 1, 1, 1, 1, 1, 
                            1, 1, 1, 1, 1, 1, 0, 0, \
                            0, 1, 1, 1, 1, 1, 1, 1, \
                            0, 0, 0, 0, 0, 1, 1, 1, \
                            1, 1, 1, 1, 1, 1, 1, 0, \
                            0, 0, 0, 0, 0, 0, 0, 0, \
                            0, 0, 0, 0, 0, 0, 0, 0, \
                            0, 0, 0, 0, 0, 0, 0, 0, \
                            0, 0, 0, 0, 0, 0, 0, 0, \
                            0, 0, 0, 0, 0, 0, 0, 0])

    print('things labels assumed to be indoor:')
    print(np.array(cat_names)[must_be_indoor==1])
    print('things labels assumed to be outdoor:')
    print(np.array(cat_names)[must_be_outdoor==1])
    print('things labels that are ambiguous:')
    print(np.array(cat_names)[(must_be_outdoor==0) & (must_be_indoor==0)])



    stuff_must_be_indoor = np.array([0, 0, 0, 0, 0, 0, 1, 0, \
                           0, 1, 1, 1, 0, 0, 0, 1, \
                           1, 1, 1, 0, 0, 0, 1, 0, \
                           0, 1, 1, 0, 0, 0, 0, 1, \
                           0, 0, 0, 0, 0, 0, 0, 0, \
                           0, 0, 0, 0, 0, 0, 0, 0, \
                           0, 1, 0, 0, 0, 0, 0, 0, \
                           0, 0, 0, 0, 1, 0, 0, 0, \
                           1, 0, 0, 0, 0, 0, 0, 0, \
                           0, 0, 0, 0, 0, 0, 0, 0, \
                           0, 0, 1, 0, 1, 0, 0, 0, \
                           1, 0, 0, 0])

    stuff_must_be_outdoor = np.array([0, 0, 1, 1, 1, 1, 0, 0, \
                            0, 0, 0, 0, 0, 0, 1, 0, \
                            0, 0, 0, 1, 0, 1, 0, 0, \
                            0, 0, 0, 0, 1, 0, 0, 0, \
                            1, 1, 0, 1, 1, 1, 0, 0, \
                            0, 0, 1, 1, 1, 0, 0, 0, \
                            1, 0, 0, 0, 0, 1, 0, 1, \
                            1, 1, 1, 1, 0, 0, 1, 1, \
                            0, 1, 1, 1, 0, 0, 0, 1, \
                            0, 0, 1, 0, 0, 1, 0, 0, \
                            0, 0, 0, 0, 0, 0, 0, 0, \
                            0, 0, 0, 0])

    print('stuff labels assumed to be indoor:')
    print(np.array(stuff_cat_names)[stuff_must_be_indoor==1])
    print('stuff labels assumed to be outdoor:')
    print(np.array(stuff_cat_names)[stuff_must_be_outdoor==1])
    print('stuff labels that are ambiguous:')
    print(np.array(stuff_cat_names)[(stuff_must_be_outdoor==0) & (stuff_must_be_indoor==0)])

    fn2load = os.path.join(default_paths.stim_labels_root, 'S%d_cocolabs_binary.csv'%subject)
    coco_df = pd.read_csv(fn2load, index_col = 0)

    cat_labels = np.array(coco_df)[:,12:92]

    indoor_columns = np.array([cat_labels[:,cc] for cc in np.where(must_be_indoor)[0]]).T
    indoor_inds_things = np.any(indoor_columns, axis=1)
    indoor_sum_things = np.sum(indoor_columns==1, axis=1)

    outdoor_columns = np.array([cat_labels[:,cc] for cc in np.where(must_be_outdoor)[0]]).T
    outdoor_inds_things = np.any(outdoor_columns, axis=1)
    outdoor_sum_things = np.sum(outdoor_columns==1, axis=1)

    np.mean(outdoor_inds_things)

    fn2load = os.path.join(default_paths.stim_labels_root, 'S%d_cocolabs_stuff_binary.csv'%subject)
    coco_stuff_df = pd.read_csv(fn2load, index_col=0)

    stuff_cat_labels = np.array(coco_stuff_df)[:,16:108]

    indoor_columns = np.array([stuff_cat_labels[:,cc] for cc in np.where(stuff_must_be_indoor)[0]]).T
    indoor_inds_stuff = np.any(indoor_columns, axis=1)
    indoor_sum_stuff = np.sum(indoor_columns==1, axis=1)

    outdoor_columns = np.array([stuff_cat_labels[:,cc] for cc in np.where(stuff_must_be_outdoor)[0]]).T
    outdoor_inds_stuff = np.any(outdoor_columns, axis=1)
    outdoor_sum_stuff = np.sum(outdoor_columns==1, axis=1)

    outdoor_all = outdoor_inds_things | outdoor_inds_stuff
    indoor_all = indoor_inds_things | indoor_inds_stuff

    ambiguous = ~outdoor_all & ~indoor_all
    conflict = outdoor_all & indoor_all

    print('\n')
    print('proportion of images that are ambiguous (no indoor or outdoor object annotation):')
    print(np.mean(ambiguous))
    print('proportion of images with conflict (have annotation for both indoor and outdoor object):')
    print(np.mean(conflict))

    conflict_indoor = conflict & (indoor_sum_things+indoor_sum_stuff > outdoor_sum_things+outdoor_sum_stuff)
    conflict_outdoor = conflict & (outdoor_sum_things+outdoor_sum_stuff > indoor_sum_things+indoor_sum_stuff)
    conflict_unresolved = conflict & (indoor_sum_things+indoor_sum_stuff == outdoor_sum_things+outdoor_sum_stuff)
    print('conflicting images that can be resolved to indoor/outdoor/tie based on number annotations:')
    print([np.mean(conflict_indoor), np.mean(conflict_outdoor), np.mean(conflict_unresolved)])

    # correct the conflicting images that we are able to resolve

    outdoor_all[conflict_indoor] = 0
    indoor_all[conflict_outdoor] = 0

    ambiguous = ~outdoor_all & ~indoor_all
    conflict = outdoor_all & indoor_all

    print('\nafter resolving conflicts based on number of annotations:')
    print('proportion of images that are ambiguous (no indoor or outdoor object annotation):')
    print(np.mean(ambiguous))
    print('proportion of images with conflict (have annotation for both indoor and outdoor object):')
    print(np.mean(conflict))


    indoor_outdoor_df = pd.DataFrame({'has_indoor': indoor_all, 'has_outdoor': outdoor_all})
    fn2save = os.path.join(default_paths.stim_labels_root, 'S%d_indoor_outdoor.csv'%subject)

    print('Saving to %s'%fn2save)
    indoor_outdoor_df.to_csv(fn2save, header=True)

    return


def write_natural_humanmade_csv(subject, which_prf_grid, debug=False):
    """
    Creating binary labels for natural/humanmade status of image patches (inferred based on presence of 
    various categories in coco and coco-stuff).
    """

    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = \
                coco_utils.get_coco_cat_info(coco_utils.coco_val)

    stuff_cat_objects, stuff_cat_names, stuff_cat_ids, stuff_supcat_names, stuff_ids_each_supcat = \
            coco_utils.get_coco_cat_info(coco_utils.coco_stuff_val) 

    must_be_natural = np.array([1,0,0,0,0,0,0,0,0,0,0,0,\
                           0,0,1,1,1,1,1,1,1,1,1,1,\
                           0,0,0,0,0,0,0,0,0,0,0,0,\
                           0,0,0,0,0,0,0,0,0,0,1,1,\
                           0,1,1,1,0,0,0,0,0,0,1,0,\
                           0,0,0,0,0,0,0,0,0,0,0,0,\
                           0,0,0,0,0,0,0,0], dtype=bool)

    must_be_humanmade = np.array([0,1,1,1,1,1,1,1,1,1,1,1,\
                             1,1,0,0,0,0,0,0,0,0,0,0,\
                             1,1,1,1,1,1,1,1,1,1,1,1,\
                             1,1,1,1,1,1,1,1,1,1,0,0,\
                             0,0,0,0,0,0,0,0,1,1,0,1,\
                             1,1,1,1,1,1,1,1,1,1,1,1,\
                             1,1,1,1,1,1,1,1], dtype=bool)

    print('things labels assumed to be natural:')
    print(np.array(cat_names)[must_be_natural==1])
    print('things labels assumed to be humanmade:')
    print(np.array(cat_names)[must_be_humanmade==1])
    print('things labels that are ambiguous:')
    print(np.array(cat_names)[(must_be_humanmade==0) & (must_be_natural==0)])

    stuff_must_be_natural = np.array([0,0,1,0,0,1,0,0,\
                  0,0,0,0,0,0,1,0,0,0,0,1, \
                  0,0,0,0,0,0,0,1,1,0,1,0, \
                  1,0,0,1,0,1,0,0,0,0,1,1, \
                  1,0,0,0,0,0,1,0,0,1,0,0, \
                  1,0,1,0,0,0,1,1,0,1,0,1, \
                  0,0,1,1,0,0,0,0,0,1,1,0, \
                  0,0,0,0,0,0,1,1,0,0,0,0], dtype=bool)

    stuff_must_be_humanmade = np.array([1,1,0,1,1,0,1,1, \
                             1,1,1,1,1,1,0,1,1,1,1,0, \
                             1,1,1,1,1,1,1,0,0,0,0,1, \
                             0,0,0,0,1,0,0,1,1,1,0,0, \
                             0,1,1,1,1,1,0,1,1,0,1,1, \
                             0,1,0,1,1,0,0,0,1,0,1,0, \
                             0,1,0,0,1,1,1,1,1,0,0,1, \
                             1,1,1,1,1,1,0,0,1,1,0,0], dtype=bool)

    print('stuff labels assumed to be natural:')
    print(np.array(stuff_cat_names)[stuff_must_be_natural==1])
    print('stuff labels assumed to be humanmade:')
    print(np.array(stuff_cat_names)[stuff_must_be_humanmade==1])
    print('stuff labels that are ambiguous:')
    print(np.array(stuff_cat_names)[(stuff_must_be_humanmade==0) & (stuff_must_be_natural==0)])

    models = initialize_fitting.get_prf_models(which_grid=which_prf_grid)    
    n_prfs = len(models)
    labels_folder = os.path.join(default_paths.stim_labels_root, 'S%d_within_prf_grid%d'%(subject, \
                                                                                        which_prf_grid))
    for prf_model_index in range(n_prfs):
             
        if debug and prf_model_index>1:
            continue
            
        fn2load = os.path.join(labels_folder, \
                              'S%d_cocolabs_binary_prf%d.csv'%(subject, prf_model_index))
        coco_df = pd.read_csv(fn2load, index_col = 0)

        cat_labels = np.array(coco_df)[:,12:92]

        natural_columns = np.array([cat_labels[:,cc] for cc in np.where(must_be_natural)[0]]).T
        natural_inds_things = np.any(natural_columns, axis=1)
        natural_sum_things = np.sum(natural_columns==1, axis=1)

        humanmade_columns = np.array([cat_labels[:,cc] for cc in np.where(must_be_humanmade)[0]]).T
        humanmade_inds_things = np.any(humanmade_columns, axis=1)
        humanmade_sum_things = np.sum(humanmade_columns==1, axis=1)

        np.mean(humanmade_inds_things)

        fn2load = os.path.join(labels_folder, \
                              'S%d_cocolabs_stuff_binary_prf%d.csv'%(subject, prf_model_index))
        coco_stuff_df = pd.read_csv(fn2load, index_col=0)

        stuff_cat_labels = np.array(coco_stuff_df)[:,16:108]

        natural_columns = np.array([stuff_cat_labels[:,cc] for cc in np.where(stuff_must_be_natural)[0]]).T
        natural_inds_stuff = np.any(natural_columns, axis=1)
        natural_sum_stuff = np.sum(natural_columns==1, axis=1)

        humanmade_columns = np.array([stuff_cat_labels[:,cc] for cc in np.where(stuff_must_be_humanmade)[0]]).T
        humanmade_inds_stuff = np.any(humanmade_columns, axis=1)
        humanmade_sum_stuff = np.sum(humanmade_columns==1, axis=1)

        humanmade_all = humanmade_inds_things | humanmade_inds_stuff
        natural_all = natural_inds_things | natural_inds_stuff
        
        ambiguous = ~humanmade_all & ~natural_all
        conflict = humanmade_all & natural_all

        print(np.mean(conflict))

        print('\n')
        print('proportion of images that are ambiguous (no natural or humanmade object annotation):')
        print(np.mean(ambiguous))
        print('proportion of images with conflict (have annotation for both natural and humanmade object):')
        print(np.mean(conflict))

        natural_humanmade_df = pd.DataFrame({'has_natural': natural_all, 'has_humanmade': humanmade_all})
        fn2save = os.path.join(labels_folder, 'S%d_natural_humanmade_prf%d.csv'%(subject, prf_model_index))

        print('Saving to %s'%fn2save)
        natural_humanmade_df.to_csv(fn2save, header=True)

    return


def write_realworldsize_csv(subject, which_prf_grid, debug=False):
    """
    Creating binary labels for real-world-size of image patches (based on approx sizes
    of object categories in coco).
    """

    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = \
                coco_utils.get_coco_cat_info(coco_utils.coco_val)

    # load a dict of sizes for each category
    fn2save = os.path.join(os.path.dirname(os.path.abspath(__file__)),'files','Realworldsize_categ.npy')
    names_to_sizes = np.load(fn2save, allow_pickle=True).item()
    # make [3 x ncateg] one-hot array of size labels for each categ
    categ_size_labels = np.array([[names_to_sizes[kk]==ii for kk in names_to_sizes.keys()] \
                                  for ii in range(3)])
    # print size lists as a sanity check
    for ii in range(3):
        print('things categories with size %d:'%ii)
        print(np.array(cat_names)[categ_size_labels[ii,:]])
        
    models = initialize_fitting.get_prf_models(which_grid=which_prf_grid)    
    n_prfs = len(models)
    labels_folder = os.path.join(default_paths.stim_labels_root, 'S%d_within_prf_grid%d'%(subject, \
                                                                                        which_prf_grid))
    for prf_model_index in range(n_prfs):
             
        if debug and prf_model_index>1:
            continue
            
        fn2load = os.path.join(labels_folder, \
                          'S%d_cocolabs_binary_prf%d.csv'%(subject, prf_model_index))
        coco_df = pd.read_csv(fn2load, index_col = 0)

        cat_labels = np.array(coco_df)[:,12:92]

        has_small = np.any(cat_labels[:,categ_size_labels[0,:]], axis=1)
        has_medium = np.any(cat_labels[:,categ_size_labels[1,:]], axis=1)
        has_large = np.any(cat_labels[:,categ_size_labels[2,:]], axis=1)

        rwsize_df = pd.DataFrame({'has_small': has_small, 'has_medium': has_medium, 'has_large': has_large})

        ambiguous = np.sum(np.array(rwsize_df), axis=1)==0
        conflict = np.sum(np.array(rwsize_df), axis=1)>1
        good = np.sum(np.array(rwsize_df), axis=1)==1

        print('proportion of images that are ambiguous (no object annotation):')
        print(np.mean(ambiguous))
        print('proportion of images with conflict (have annotation for multiple sizes of objects):')
        print(np.mean(conflict))
        print('proportion of images unambiguous (exactly one size label):')
        print(np.mean(good))

        fn2save = os.path.join(labels_folder, 'S%d_realworldsize_prf%d.csv'%(subject, prf_model_index))

        print('Saving to %s'%fn2save)
        rwsize_df.to_csv(fn2save, header=True)

    return

def count_labels_each_prf(which_prf_grid=5, debug=False):

    """
    Load category labels for each set of images, count how many of each unique 
    label value exist in the set. Counting for each subject's set of
    10000 coco images, and separately for the trialwise sequence of viewed images 
    (i.e. including repeats in the count)
    """
        
    fn2save = os.path.join(default_paths.stim_labels_root, 'Coco_label_counts_all_prf_grid%d.npy'%(which_prf_grid))   
    
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = \
            coco_utils.get_coco_cat_info(coco_utils.coco_val)

    stuff_cat_objects, stuff_cat_names, stuff_cat_ids, stuff_supcat_names, stuff_ids_each_supcat = \
            coco_utils.get_coco_cat_info(coco_utils.coco_stuff_val)

    models = initialize_fitting.get_prf_models(which_grid=which_prf_grid)    
    n_prfs = models.shape[0]
    n_subjects=8
    
    n_supcat = len(supcat_names); n_cat = len(cat_names)
    n_stuff_supcat = len(stuff_supcat_names); n_stuff_cat = len(stuff_cat_names)
    
    # arrays for counts over each subject's list of 10000 images
    things_supcat_counts = np.zeros((n_subjects, n_prfs, n_supcat))
    things_cat_counts = np.zeros((n_subjects, n_prfs, n_cat)) 
    
    stuff_supcat_counts = np.zeros((n_subjects, n_prfs, n_stuff_supcat))
    stuff_cat_counts = np.zeros((n_subjects, n_prfs, n_stuff_cat))
    
    # arrays of how many actual trials the label occured on
    things_supcat_counts_trntrials = np.zeros((n_subjects, n_prfs, n_supcat))
    things_cat_counts_trntrials = np.zeros((n_subjects, n_prfs, n_cat))
    things_supcat_counts_valtrials = np.zeros((n_subjects, n_prfs, n_supcat))
    things_cat_counts_valtrials = np.zeros((n_subjects, n_prfs, n_cat))
    
    stuff_supcat_counts_trntrials = np.zeros((n_subjects, n_prfs, n_stuff_supcat))
    stuff_cat_counts_trntrials = np.zeros((n_subjects, n_prfs, n_stuff_cat))
    stuff_supcat_counts_valtrials = np.zeros((n_subjects, n_prfs, n_stuff_supcat))
    stuff_cat_counts_valtrials = np.zeros((n_subjects, n_prfs, n_stuff_cat))

    if debug:
        subjects = [1]
    else:
        subjects = np.arange(1,9)
        
    for si, ss in enumerate(subjects):
        labels_folder = os.path.join(default_paths.stim_labels_root, \
                                         'S%d_within_prf_grid%d'%(ss, which_prf_grid))
        print('loading labels from folder %s...'%labels_folder)
        
        # now get actual trial sequence for the this subject
        trial_order = nsd_utils.get_master_image_order()
        session_inds = nsd_utils.get_session_inds_full()  
        # remove any trials that weren't actually shown to this subject
        sessions = np.arange(0, np.min([40, nsd_utils.max_sess_each_subj[si]]))
        print('subject %d has sessions up to %d'%(ss,nsd_utils.max_sess_each_subj[si]))
        trial_order = trial_order[np.isin(session_inds, sessions)]
        assert(len(trial_order)/nsd_utils.trials_per_sess==nsd_utils.max_sess_each_subj[si])
        trial_order_val = trial_order[trial_order<1000]
        trial_order_trn = trial_order[trial_order>=1000]
        print('num training/val/total images actually shown: %d/%d/%d'\
              %(len(trial_order_trn), len(trial_order_val), len(trial_order)))
        
        sys.stdout.flush()
        for prf_model_index in range(n_prfs):
            if debug and prf_model_index>1:
                continue
            coco_things_labels_fn = os.path.join(labels_folder, \
                                      'S%d_cocolabs_binary_prf%d.csv'%(ss, prf_model_index))  
            coco_stuff_labels_fn = os.path.join(labels_folder, \
                                      'S%d_cocolabs_stuff_binary_prf%d.csv'%(ss, prf_model_index))  
            coco_things_df = pd.read_csv(coco_things_labels_fn, index_col=0)
            coco_stuff_df = pd.read_csv(coco_stuff_labels_fn, index_col=0)

            things_supcat_labels = np.array(coco_things_df)[:,0:12]
            things_cat_labels = np.array(coco_things_df)[:,12:92]
            stuff_supcat_labels = np.array(coco_stuff_df)[:,0:16]
            stuff_cat_labels = np.array(coco_stuff_df)[:,16:108]

            things_supcat_counts[si,prf_model_index,:] = np.sum(things_supcat_labels, axis=0)            
            things_cat_counts[si,prf_model_index,:] = np.sum(things_cat_labels, axis=0)
            stuff_supcat_counts[si,prf_model_index,:] = np.sum(stuff_supcat_labels, axis=0)
            stuff_cat_counts[si,prf_model_index,:] = np.sum(stuff_cat_labels, axis=0)

            things_supcat_counts_trntrials[si,prf_model_index,:] = np.sum(things_supcat_labels[trial_order_trn,:], axis=0)           
            things_cat_counts_trntrials[si,prf_model_index,:] = np.sum(things_cat_labels[trial_order_trn,:], axis=0)
            stuff_supcat_counts_trntrials[si,prf_model_index,:] = np.sum(stuff_supcat_labels[trial_order_trn,:], axis=0)
            stuff_cat_counts_trntrials[si,prf_model_index,:] = np.sum(stuff_cat_labels[trial_order_trn,:], axis=0)

            things_supcat_counts_valtrials[si,prf_model_index,:] = np.sum(things_supcat_labels[trial_order_val,:], axis=0)           
            things_cat_counts_valtrials[si,prf_model_index,:] = np.sum(things_cat_labels[trial_order_val,:], axis=0)
            stuff_supcat_counts_valtrials[si,prf_model_index,:] = np.sum(stuff_supcat_labels[trial_order_val,:], axis=0)
            stuff_cat_counts_valtrials[si,prf_model_index,:] = np.sum(stuff_cat_labels[trial_order_val,:], axis=0)

            
    print('Saving to %s'%fn2save)
    np.save(fn2save, {'things_supcat_counts': things_supcat_counts,\
                      'things_cat_counts': things_cat_counts, \
                      'stuff_supcat_counts': stuff_supcat_counts, \
                      'stuff_cat_counts': stuff_cat_counts, \
                      'things_supcat_counts_trntrials': things_supcat_counts_trntrials,\
                      'things_cat_counts_trntrials': things_cat_counts_trntrials, \
                      'stuff_supcat_counts_trntrials': stuff_supcat_counts_trntrials, \
                      'stuff_cat_counts_trntrials': stuff_cat_counts_trntrials, \
                      'things_supcat_counts_valtrials': things_supcat_counts_valtrials,\
                      'things_cat_counts_valtrials': things_cat_counts_valtrials, \
                      'stuff_supcat_counts_valtrials': stuff_supcat_counts_valtrials, \
                      'stuff_cat_counts_valtrials': stuff_cat_counts_valtrials}, allow_pickle=True)

def get_top_two_subcateg(which_prf_grid=5):
    
    """
    Choose the top two most common subordinate categories for every super-category, 
    will be used to test subordinate-level discriminability.
    """
    
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = \
            coco_utils.get_coco_cat_info(coco_utils.coco_val)

    stuff_cat_objects, stuff_cat_names, stuff_cat_ids, stuff_supcat_names, stuff_ids_each_supcat = \
            coco_utils.get_coco_cat_info(coco_utils.coco_stuff_val)


    labels_folder = os.path.join(default_paths.stim_labels_root)
    fn2load = os.path.join(default_paths.stim_labels_root, \
                           'Coco_label_counts_all_prf_grid%d.npy'%(which_prf_grid))      
    out = np.load(fn2load, allow_pickle=True).item()
    fn2save = os.path.join(default_paths.stim_labels_root, \
                           'Coco_supcat_top_two_prf_grid%d.npy'%(which_prf_grid)) 
    
    things_top_two = {}

    counts = out['things_cat_counts']
    counts_total = np.sum(np.sum(counts, axis=0), axis=0)
    for si, scname in enumerate(supcat_names):

        cat_inds_this_supcat = np.where(np.isin(cat_ids, ids_each_supcat[si]))[0]

        subcat_names = np.array(cat_names)[cat_inds_this_supcat]

        if len(cat_inds_this_supcat)<2:
            print('skip %s, only one subcategory'%scname)
            things_top_two[scname] = []
            continue

        subcat_counts = counts_total[cat_inds_this_supcat]
        top_ranked = np.flip(np.argsort(subcat_counts))

        things_top_two[scname] = [subcat_names[ii] for ii in top_ranked[0:2]]

        print('\nsuper-ordinate category %s, top sub-categories are:'%scname)
        print(np.array(subcat_names)[top_ranked])
        subcat_means = np.mean(np.mean(counts[:,:,cat_inds_this_supcat], axis=0), axis=0)
        print('average n occurences out of 10000 ims (avg over all pRFs):')
        print(subcat_means[top_ranked].round(0))
        subcat_mins = np.min(np.min(counts[:,:,cat_inds_this_supcat], axis=0), axis=0)
        print('min n occurences out of 10000 ims (in any pRF):')
        print(subcat_mins[top_ranked])
        
    stuff_top_two = {}

    counts = out['stuff_cat_counts']
    counts_total = np.sum(np.sum(counts, axis=0), axis=0)
    for si, scname in enumerate(stuff_supcat_names):

        cat_inds_this_supcat = np.where(np.isin(stuff_cat_ids, stuff_ids_each_supcat[si]))[0]

        subcat_names = np.array(stuff_cat_names)[cat_inds_this_supcat]

        # excluding any sub-category here with the "other" label,
        # because not specific enough
        inds_include = ['other' not in n for n in subcat_names]
        cat_inds_this_supcat = cat_inds_this_supcat[inds_include]
        subcat_names = subcat_names[inds_include]

        if len(cat_inds_this_supcat)<2:
            print('skip %s, only one subcategory'%scname)
            stuff_top_two[scname] = []
            continue

        subcat_counts = counts_total[cat_inds_this_supcat]
        top_ranked = np.flip(np.argsort(subcat_counts))
        if scname=='ground':
            # skipping the second sub-category here because "road" and "pavement" 
            # are too similar.
            stuff_top_two[scname] = [subcat_names[ii] for ii in top_ranked[[0,2]]]        
        else:
            stuff_top_two[scname] = [subcat_names[ii] for ii in top_ranked[0:2]]

        print('\nsuper-ordinate category %s, top ranked sub-categories are:'%scname)
        print(np.array(subcat_names)[top_ranked])
        subcat_means = np.mean(np.mean(counts[:,:,cat_inds_this_supcat], axis=0), axis=0)
        print('average n occurences out of 10000 ims (avg over all pRFs):')
        print(subcat_means[top_ranked].round(0))
        subcat_mins = np.min(np.min(counts[:,:,cat_inds_this_supcat], axis=0), axis=0)
        print('min n occurences out of 10000 ims (in any pRF):')
        print(subcat_mins[top_ranked])
        
    print('saving to %s'%fn2save)
    np.save(fn2save, {'things_top_two': things_top_two, \
                     'stuff_top_two': stuff_top_two})
    
    
    
def concat_labels_each_prf(subject, which_prf_grid, verbose=False, debug=False):

    """
    Concatenate the csv files containing a range of different labels for each image patch.
    Save a file that is loaded later on by "load_labels_each_prf".
    Each column is a single attribute with multiple levels - nans indicate that the attribute 
    was ambiguous in that pRF.
    """

    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = \
            coco_utils.get_coco_cat_info(coco_utils.coco_val)

    stuff_cat_objects, stuff_cat_names, stuff_cat_ids, stuff_supcat_names, stuff_ids_each_supcat = \
            coco_utils.get_coco_cat_info(coco_utils.coco_stuff_val)

    labels_folder = os.path.join(default_paths.stim_labels_root, \
                                     'S%d_within_prf_grid%d'%(subject, which_prf_grid))
    # this file will get made once every time this method is run, it is same for all subjects/pRFs.
    save_name_groups = os.path.join(default_paths.stim_labels_root,'All_concat_labelgroupnames.npy')

    print('concatenating labels from folders at %s and %s (will be slow...)'%\
          (default_paths.stim_labels_root, labels_folder))
    sys.stdout.flush()
    
    top_two_fn = os.path.join(default_paths.stim_labels_root, \
                           'Coco_supcat_top_two_prf_grid%d.npy'%(which_prf_grid))
    top_two = np.load(top_two_fn, allow_pickle=True).item()
    
    # list all the attributes we want to look at here
    discrim_type_list = ['indoor_outdoor','natural_humanmade','animacy','real_world_size_binary',\
                         'real_world_size_continuous']

    discrim_type_list+=supcat_names # presence or absence of each superordinate category
    # within each superordinate category, will label the basic-level sub-categories 
    for scname in supcat_names:
        if len(top_two['things_top_two'][scname])>0:           
            discrim_type_list+=['within_%s'%scname]

    # same idea for the coco-stuff labels
    discrim_type_list+=stuff_supcat_names
    for scname in stuff_supcat_names:
        if len(top_two['stuff_top_two'][scname])>0:           
            discrim_type_list+=['within_%s'%scname]

    n_sem_axes = len(discrim_type_list)
    n_trials = 10000
    models = initialize_fitting.get_prf_models(which_grid=which_prf_grid)    
    n_prfs = models.shape[0]

    # load the labels for indoor vs outdoor (which is defined across entire images, not within pRF)
    in_out_labels_fn = os.path.join(default_paths.stim_labels_root, 'S%d_indoor_outdoor.csv'%subject)
    if verbose:
        print('Loading pre-computed features from %s'%in_out_labels_fn)
    in_out_df = pd.read_csv(in_out_labels_fn, index_col=0)

    for prf_model_index in range(n_prfs):

        if debug and prf_model_index>1:
            continue
            
        # will make a single dataframe for each pRF
        labels_df = pd.DataFrame({})
        save_name = os.path.join(labels_folder, \
                                  'S%d_concat_prf%d.csv'%(subject, prf_model_index))
        
        # load all the different binary label sets for this pRF
        nat_hum_labels_fn = os.path.join(labels_folder, \
                                  'S%d_natural_humanmade_prf%d.csv'%(subject, prf_model_index))
        size_labels_fn = os.path.join(labels_folder, \
                                  'S%d_realworldsize_prf%d.csv'%(subject, prf_model_index))
        coco_things_labels_fn = os.path.join(labels_folder, \
                                  'S%d_cocolabs_binary_prf%d.csv'%(subject, prf_model_index))  
        coco_stuff_labels_fn = os.path.join(labels_folder, \
                                  'S%d_cocolabs_stuff_binary_prf%d.csv'%(subject, prf_model_index))  

        if verbose and (prf_model_index==0):
            print('Loading pre-computed features from %s'%nat_hum_labels_fn)
            print('Loading pre-computed features from %s'%size_labels_fn)
            print('Loading pre-computed features from %s'%coco_things_labels_fn)
            print('Loading pre-computed features from %s'%coco_stuff_labels_fn)

        nat_hum_df = pd.read_csv(nat_hum_labels_fn, index_col=0)  
        size_df = pd.read_csv(size_labels_fn, index_col=0)  
        coco_things_df = pd.read_csv(coco_things_labels_fn, index_col=0)
        coco_stuff_df = pd.read_csv(coco_stuff_labels_fn, index_col=0)

        col_names_all = []
        for dd in range(n_sem_axes):

            labels=None; colnames=None;
            
            discrim_type = discrim_type_list[dd]
            if 'indoor_outdoor' in discrim_type:
                labels = np.array(in_out_df).astype(np.float32)
                colnames = list(in_out_df.keys())

            elif 'natural_humanmade' in discrim_type:
                labels = np.array(nat_hum_df).astype(np.float32)
                colnames = list(nat_hum_df.keys())
            
            elif 'real_world_size' in discrim_type:
                if 'binary' in discrim_type:
                    # use just small and big, ignoring medium.
                    labels = np.array(size_df)[:,[0,2]].astype(np.float32)
                    colnames = [size_df.keys()[0], size_df.keys()[2]]
                elif 'continuous' in discrim_type:
                    # use all three levels, for a continuous variable.
                    labels = np.array(size_df).astype(np.float32)
                    colnames = list(size_df.keys())
                    
            elif 'animacy' in discrim_type:
                supcat_labels = np.array(coco_things_df)[:,0:12]
                animate_supcats = [1,9]
                inanimate_supcats = [ii for ii in range(12)\
                                     if ii not in animate_supcats]
                has_animate = np.any(np.array([supcat_labels[:,ii]==1 \
                                               for ii in animate_supcats]), axis=0)
                has_inanimate = np.any(np.array([supcat_labels[:,ii]==1 \
                                            for ii in inanimate_supcats]), axis=0)
                labels = np.concatenate([has_animate[:,np.newaxis], \
                                             has_inanimate[:,np.newaxis]], axis=1).astype(np.float32)
                colnames = ['has_animate','has_inanimate']

            elif discrim_type in supcat_names:
                has_label = np.any(np.array(coco_things_df)[:,0:12]==1, axis=1)
                label1 = np.array(coco_things_df[discrim_type])[:,np.newaxis]
                label2 = (label1==0) & (has_label[:,np.newaxis])
                labels = np.concatenate([label1, label2], axis=1).astype(np.float32)
                colnames = ['has_%s'%discrim_type, 'has_other']

            elif discrim_type in stuff_supcat_names:
                has_label = np.any(np.array(coco_stuff_df)[:,0:16]==1, axis=1)
                label1 = np.array(coco_stuff_df[discrim_type])[:,np.newaxis]
                label2 = (label1==0) & (has_label[:,np.newaxis])
                labels = np.concatenate([label1, label2], axis=1).astype(np.float32)
                colnames = ['has_%s'%discrim_type, 'has_other']

            elif 'within_' in discrim_type:

                supcat_name = discrim_type.split('within_')[1]
                # for a given super-category, label the individual sub-categories.
                if supcat_name in supcat_names:
                    subcats_use = top_two['things_top_two'][supcat_name]
                    labels = np.array([np.array(coco_things_df[subcats_use[0]]), \
                             np.array(coco_things_df[subcats_use[1]])]).astype(np.float32).T
                    colnames = subcats_use

                elif supcat_name in stuff_supcat_names:
                    subcats_use = top_two['stuff_top_two'][supcat_name]
                    labels = np.array([np.array(coco_stuff_df[subcats_use[0]]), \
                             np.array(coco_stuff_df[subcats_use[1]])]).astype(np.float32).T
                    colnames = subcats_use
            
            col_names_all.append(colnames)
            
            if verbose:
                print(discrim_type)
                print(colnames)          
                if labels.shape[1]==2:
                    print('num 1/1, 1/0, 0/1, 0/0:')
                    print([np.sum((labels[:,0]==1) & (labels[:,1]==1)), \
                    np.sum((labels[:,0]==1) & (labels[:,1]==0)),\
                    np.sum((labels[:,0]==0) & (labels[:,1]==1)),\
                    np.sum((labels[:,0]==0) & (labels[:,1]==0))])
                else:
                    print('num each column:')
                    print(np.sum(labels, axis=0).astype(int))

            # remove any images with >1 or 0 labels, since these are ambiguous.
            assert(len(colnames)==labels.shape[1])
            has_one_label = np.sum(labels, axis=1)==1
            labels = np.array([np.where(labels[ii,:]==1)[0][0] \
                       if has_one_label[ii] else np.nan \
                       for ii in range(labels.shape[0])])

            if verbose:
                print('n trials labeled/ambiguous: %d/%d'%\
                      (np.sum(~np.isnan(labels)), np.sum(np.isnan(labels))))

            labels_df[discrim_type] = labels

        if verbose:
            print('Saving to %s'%save_name)

        labels_df.to_csv(save_name, header=True)
        
        if prf_model_index==0:
            print('Saving to %s'%(save_name_groups))
            np.save(save_name_groups, {'discrim_type_list': discrim_type_list, \
                                      'col_names_all': col_names_all}, allow_pickle=True)
        
def count_highlevel_labels(which_prf_grid=5, axes_to_do=[0,2,3], \
                           debug=False):

    """
    Count occurences of each high-level semantic label in the entire image set
    for each NSD participant.
    """

    models = initialize_fitting.get_prf_models(which_prf_grid)
    n_prfs = models.shape[0]

   
    subjects = np.array(list(np.arange(1,9)) + [999])
    n_subjects = len(subjects)
    n_axes = len(axes_to_do)
    n_levels = 3; # levels are [label1, label2, ambiguous]
    
    counts = np.zeros((n_subjects, n_prfs, n_axes, n_levels))
    
    for si, ss in enumerate(subjects):
    
        if debug and si>0:
            continue
            
        labels_folder = os.path.join(default_paths.stim_labels_root, \
                                     'S%d_within_prf_grid%d'%(ss, which_prf_grid))
        if si==0:
            groups = np.load(os.path.join(default_paths.stim_labels_root,\
                                  'All_concat_labelgroupnames.npy'), allow_pickle=True).item()
            group_names = [groups['col_names_all'][aa] for aa in axes_to_do]
            axis_names = [groups['discrim_type_list'][aa] for aa in axes_to_do]
          
        print('loading labels from folders at %s (will be slow...)'%(labels_folder))
  
        # figure out what images are available for this subject - assume that we 
        # will be using all the available sessions. 
        image_order = nsd_utils.get_master_image_order()    
        session_inds = nsd_utils.get_session_inds_full()
        if ss!=999:
            sessions = np.arange(nsd_utils.max_sess_each_subj[ss-1])
            inds2use = np.isin(session_inds, sessions) # remove any sessions that weren't shown
            # list of all the image indices shown on each trial
            image_order = image_order[inds2use] 
        # reduce to the ~10,000 unique images
        image_order = np.unique(image_order) 
        print('analyzing counts for S%d, %d images'%(ss, len(image_order)))
              
        n_trials = image_order.shape[0]

        for prf_model_index in range(n_prfs):

            if debug and prf_model_index>1:
                continue

            fn2load = os.path.join(labels_folder, \
                                      'S%d_concat_prf%d.csv'%(ss, prf_model_index))
            concat_df = pd.read_csv(fn2load, index_col=0)
            labels = np.array(concat_df)
            labels = labels[image_order,:]
           
            for ai, aa in enumerate(axes_to_do):
                counts[si, prf_model_index, ai, 0] = np.sum(labels[:,aa]==0)
                counts[si, prf_model_index, ai, 1] = np.sum(labels[:,aa]==1)
                counts[si, prf_model_index, ai, 2] = np.sum(np.isnan(labels[:,aa]))
            
        if not debug:
            assert(np.all(np.sum(counts[si,:,:,:], axis=2)==n_trials))
              
                
    fn2save = os.path.join(default_paths.stim_labels_root, 'Highlevel_counts_all.npy')
    print('saving to %s'%fn2save)
    np.save(fn2save, {'counts': counts, \
                     'group_names': group_names, \
                     'axis_names': axis_names}, 
            allow_pickle=True)

    return 