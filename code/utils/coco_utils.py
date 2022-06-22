# import basic modules
import sys
import os
import numpy as np
import pandas as pd
import PIL
import h5py
from ast import literal_eval

from utils import default_paths, nsd_utils, segmentation_utils

# import coco api tools
sys.path.append(os.path.join(default_paths.coco_api_path,'cocoapi','PythonAPI'))
from pycocotools.coco import COCO

def init_coco(coco_api_path=None):
    
    if coco_api_path is None:
        coco_api_path = default_paths.coco_api_path
        
    annot_file_val = os.path.join(coco_api_path, 'annotations','instances_val2017.json')
    coco_val = COCO(annot_file_val)
    annot_file_trn = os.path.join(coco_api_path, 'annotations', 'instances_train2017.json')
    coco_trn = COCO(annot_file_trn)
    
    return coco_trn, coco_val

print('Initializing coco api...')
coco_trn, coco_val = init_coco()

def init_coco_stuff(coco_api_path=None):
    
    if coco_api_path is None:
        coco_api_path = default_paths.coco_api_path
        
    annot_file_val = os.path.join(coco_api_path, 'annotations','stuff_val2017.json')
    coco_stuff_val = COCO(annot_file_val)
    annot_file_trn = os.path.join(coco_api_path, 'annotations', 'stuff_train2017.json')
    coco_stuff_trn = COCO(annot_file_trn)
    
    return coco_stuff_trn, coco_stuff_val

print('Initializing coco api...')
coco_stuff_trn, coco_stuff_val = init_coco_stuff()

def get_coco_cat_info(coco_object=None):
    
    """ 
    Get lists of all the category names, ids, supercategories in COCO.
    """
    
    if coco_object is None:
        coco_object = coco_val
        
    cat_objects = coco_object.loadCats(coco_object.getCatIds())
    cat_names=[cat['name'] for cat in cat_objects]   
    cat_ids=[cat['id'] for cat in cat_objects]

    supcat_names = list(set([cat['supercategory'] for cat in cat_objects]))
    supcat_names.sort()

    ids_each_supcat = []
    for sc in range(len(supcat_names)):
        this_supcat = [supcat_names[sc]==cat['supercategory'] for cat in cat_objects]
        ids = [cat_objects[ii]['id'] for ii in range(len(cat_names)) if this_supcat[ii]==True]
        ids_each_supcat.append(ids)

    return cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat
   
def get_ims_in_supcat(supcat_name, coco_ids, stuff=False):
    
    """
    For a given supercategory name, find all the images in 'coco_ids' that include an annotation of that super-category.
    Return boolean array same size as coco_ids.
    """
    if stuff:
        coco_v = coco_stuff_val
        coco_t = coco_stuff_trn
    else:
        coco_v = coco_val
        coco_t = coco_trn
       
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = get_coco_cat_info(coco_v)
    sc_ind = [ii for ii in range(len(supcat_names)) if supcat_names[ii]==supcat_name]
    assert(len(sc_ind)==1)
    sc_ind=sc_ind[0]
    all_ims_in_supcat_val = np.concatenate([coco_v.getImgIds(catIds = cid) for cid in ids_each_supcat[sc_ind]], axis=0);
    all_ims_in_supcat_trn = np.concatenate([coco_t.getImgIds(catIds = cid) for cid in ids_each_supcat[sc_ind]], axis=0);
    all_ims_in_supcat = np.concatenate((all_ims_in_supcat_val, all_ims_in_supcat_trn), axis=0)
    
    ims_in_supcat = np.isin(coco_ids, all_ims_in_supcat)
    
    return np.squeeze(ims_in_supcat)

def list_supcats_each_image(coco_ids, stuff=False):
    
    """
    For all the different super-categories, list which images in coco_ids
    contain an instance of that super-category. 
    Returns a binary matrix, and also a list of which super-cats are in each image.
    """
    
    ims_each_supcat = []
    if stuff:
        coco_object = coco_stuff_val
    else:
        coco_object = coco_val
        
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = get_coco_cat_info(coco_object)

    for sc, scname in enumerate(supcat_names):
        ims_in_supcat = get_ims_in_supcat(scname, coco_ids, stuff=stuff)
        ims_each_supcat.append(ims_in_supcat)
        
    ims_each_supcat = np.array(ims_each_supcat)
    supcats_each_image = [np.where(ims_each_supcat[:,ii])[0] for ii in range(ims_each_supcat.shape[1])]
    ims_each_supcat = ims_each_supcat.astype('int').T
    
    return ims_each_supcat, supcats_each_image

def get_ims_in_cat(cat_name, coco_ids, stuff=False):
    
    """
    For a given category name, find all the images in 'coco_ids' that include an annotation of that super-category.
    Return boolean array same size as coco_ids.
    """
    if stuff:
        coco_v = coco_stuff_val
        coco_t = coco_stuff_trn
    else:
        coco_v = coco_val
        coco_t = coco_trn
        
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = get_coco_cat_info(coco_v)
    cid = [ii for ii in range(len(cat_names)) if cat_names[ii]==cat_name]
    assert(len(cid)==1)
    cid = cid[0]
    all_ims_in_cat_val = coco_v.getImgIds(catIds = cat_ids[cid])
    all_ims_in_cat_trn = coco_t.getImgIds(catIds = cat_ids[cid])
    all_ims_in_cat = np.concatenate((all_ims_in_cat_val, all_ims_in_cat_trn), axis=0)
    
    ims_in_cat = np.isin(coco_ids, all_ims_in_cat)
    
    return np.squeeze(ims_in_cat)

def list_cats_each_image(coco_ids, stuff=False):
    
    """
    For all the different categories, list which images in coco_ids
    contain an instance of that category. 
    Returns a binary matrix, and also a list of which categories are in each image.
    """
        
    ims_each_cat = []
    if stuff:
        coco_object = coco_stuff_val
    else:
        coco_object = coco_val
    cat_objects, cat_names, cat_ids, supcat_names, ids_each_supcat = get_coco_cat_info(coco_object)

    for cc, cname in enumerate(cat_names):
        ims_in_cat = get_ims_in_cat(cname, coco_ids, stuff=stuff)
        ims_each_cat.append(ims_in_cat)
        
    ims_each_cat = np.array(ims_each_cat)
    cats_each_image = [np.where(ims_each_cat[:,ii])[0] for ii in range(ims_each_cat.shape[1])]
    ims_each_cat = ims_each_cat.astype('int').T
    
    return ims_each_cat, cats_each_image

def get_coco_ids_indep(n_images = 10000):

    # choosing a set of images from the COCO dataset which are 
    # NOT overlapping with any of the NSD images.

    # first listing all the files in training image dir
    files = os.listdir(os.path.join(default_paths.coco_ims_path, 'train2017'))
    cocotrnids = np.array([int(ff.split('.jpg')[0]) for ff in files])

    # find all the cocoids used in NSD
    ids_nsd = [] 
    for ss in np.arange(1,9):
        subj_df = nsd_utils.get_subj_df(ss)
        ids_nsd += list(subj_df['cocoId'])

    ids_nsd = np.unique(np.array(ids_nsd))

    # choose a set of images that won't overlap
    ids_indep = cocotrnids[~np.isin(cocotrnids, ids_nsd)]
    rndseed = 934593 # hard code this so we always get same sequence
    np.random.seed(rndseed)
    ids_indep = np.random.choice(ids_indep, n_images, replace=False)
    assert(not any(np.isin(ids_nsd, ids_indep)))
    assert(not any(np.isin(ids_indep, ids_nsd)))

    # save as a file for use later
    ims_df = pd.DataFrame(data=ids_use_indep, columns=['cocoId'])
    ims_df.to_csv(os.path.join(default_paths.root, 'features', 'coco_ids_indep_set.csv'))
    
def prep_indep_coco_images(n_pix=240, debug=False):

    # makes Indep_set_stimuli_240.h5py
    # brick of images for the independent image set, COCO ims that are not in NSD
    
    # load this file that indicates which coco images we will use here
    ids = pd.read_csv(os.path.join(default_paths.root, 'features', 'coco_ids_indep_set.csv'), index_col=0)
    coco_ids = np.array(ids['cocoId'])
    n_images = len(coco_ids)
    
    bboxes = []
    coco_split = []
    filenames = []
                 
    coco_image_stack = np.zeros((n_images, 3, n_pix, n_pix))
    
    for ii, coco_id in enumerate(coco_ids):
        
        if debug and ii>1:
            bboxes += [[]]
            coco_split += ['train2017']
            filenames += [[]]
            continue
        
        cocoim_raw = os.path.join(default_paths.coco_ims_path, 'train2017', '%012d.jpg'%coco_id)
        print('%d of %d'%(ii, len(coco_ids)))
        print('loading from %s'%cocoim_raw)
        sys.stdout.flush()
        coco_image = PIL.Image.open(cocoim_raw)
        
        # preprocess these images to be square, black and white
        # process at same size as i processed the NSD images
        cropped, bbox = segmentation_utils.crop_to_square(np.array(coco_image))
        resized = PIL.Image.fromarray(cropped).resize([n_pix, n_pix], \
                                                       resample=PIL.Image.BILINEAR)
        resized = np.array(resized.convert('RGB'))

        # make sure this is now an RGB image, uint8 values spanning 0-255
        assert(resized.shape[2]==3)
        assert(resized.dtype=='uint8')
        coco_image_stack[ii,:,:,:] = np.moveaxis(resized, [0,1,2], [1,2,0])
        
        # making these to match subject df from NSD, makes processing easier later on
        bboxes += [tuple(bbox)]
        coco_split += ['train2017']
        filenames += [cocoim_raw]
        
    info_df = pd.DataFrame({'cocoId': coco_ids, 'cropBox': bboxes, 'cocoSplit': coco_split, 'filename_raw': filenames})
    fn2save_df = os.path.join(default_paths.stim_root, 'Indep_set_info.csv')
    
    print('saving to %s'%fn2save_df)
    info_df.to_csv(fn2save_df)
    
    if debug:
        fn2save = os.path.join(default_paths.stim_root, 'Indep_set_stimuli_%d_debug.h5py'%n_pix)
    else:
        fn2save = os.path.join(default_paths.stim_root, 'Indep_set_stimuli_%d.h5py'%n_pix)
     
    print('saving to %s'%fn2save)
    
    with h5py.File(fn2save, 'w') as hf:
        key = 'stimuli'
        val = coco_image_stack  
        hf.create_dataset(key,data=val, dtype='uint8')
        
    print('done')

def load_indep_coco_images(n_pix=240):
    
    ims_fn = os.path.join(default_paths.stim_root, 'Indep_set_stimuli_%d.h5py'%n_pix)
    print('\nloading images from %s\n'%ims_fn)
    image_data = nsd_utils.load_from_hdf5(ims_fn)
    
    return image_data

def load_indep_coco_info():
    
    info_fn = os.path.join(default_paths.stim_root, 'Indep_set_info.csv')
    print('\nloading image info from %s\n'%info_fn)
    info_df = pd.read_csv(info_fn, index_col=0)
    
    info_df['cropBox'] = info_df['cropBox'].apply(literal_eval)
    
    return info_df