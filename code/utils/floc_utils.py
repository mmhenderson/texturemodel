import numpy as np
import os, sys
import pandas as pd
import time, h5py
import PIL.Image

from utils import default_paths, nsd_utils

# code to work with stimuli from the fLoc category localizer (from Grill-Spector lab)
# access raw stimuli here:
# http://vpnl.stanford.edu/fLoc/

categories = ['word','number',
              'body','limb',
              'adult','child',
              'corridor','house',
              'car','instrument',
              'food1','food2']
# the food domain is also added on here, using stims from Jain et al. 2022
# note that there are no meaningful "categories" within the food domain, 
# the "food1" and "food2" labels above are just first half and second 
# half of all our food images.

domains = ['characters','bodies','faces','places','objects','food']

n_each_nonfood_categ = 144;
n_nonfood_categ = 10;
n_each_food_categ = 41;
n_food_categ = 2;

floc_image_root = default_paths.floc_image_root
food_image_root = default_paths.food_image_root


def prep_images(newsize=(240,240)):
    
    n_images = n_each_nonfood_categ * n_nonfood_categ + n_each_food_categ * n_food_categ
    ims_brick = np.zeros((n_images, 1, newsize[0], newsize[1]))

    categ_list = []
    domain_list = []
    instance_list = []
    filename_list = []

    count = -1;

    for cc, categ in enumerate(categories):

        print('processing %s'%categ)
        dd = int(np.floor(cc/2))
        
        if 'food' in categ:
            nc = n_each_food_categ
        else:
            nc = n_each_nonfood_categ
            
        for ii in np.arange(1,nc+1):

            count += 1

            if categ=='food1':
                filename = os.path.join(food_image_root,'%s-%d.jpg'%('food', ii))
            elif categ=='food2':
                filename = os.path.join(food_image_root,'%s-%d.jpg'%('food', ii+n_each_food_categ))
            else:
                filename = os.path.join(floc_image_root,'%s-%d.jpg'%(categ, ii))

            if ii==1:       
                print('loading from %s'%filename)
            im = PIL.Image.open(filename)
            
            if im.mode=='RGB':
                # take just first channel (each channel is a duplicate)
                im = PIL.Image.fromarray(np.asarray(im)[:,:,0])

            im_resized = im.resize(newsize, resample=PIL.Image.ANTIALIAS)

            ims_brick[count,0,:,:] = im_resized

            categ_list += [categ]
            domain_list += [domains[dd]]
            instance_list += [ii]
            filename_list += [filename]
            
    
    floc_labels = pd.DataFrame({'category': categ_list, \
                                'domain': domain_list, \
                                'instance': instance_list, \
                                'filename': filename_list})

    csv_filename =  os.path.join(floc_image_root,'floc_image_labels.csv')

    print('Writing csv labels to %s'%csv_filename)
    floc_labels.to_csv(csv_filename, index=False)
        
    
    fn2save = os.path.join(floc_image_root,'floc_stimuli_%d.h5py'%newsize[0])

    print('Writing resized images to %s\n'%fn2save)

    t = time.time()
    with h5py.File(fn2save, 'w') as data_set:
        dset = data_set.create_dataset("stimuli", np.shape(ims_brick), dtype=np.uint8)
        data_set['/stimuli'][:,:,:] = ims_brick
        data_set.close()  
    elapsed = time.time() - t

    print('Took %.5f sec to write file'%elapsed)

    return

def load_floc_images(npix=240):
    
    image_filename = os.path.join(floc_image_root, 'floc_stimuli_%d.h5py'%npix)
    print('loading images from %s'%image_filename)
    ims = nsd_utils.load_from_hdf5(image_filename)

    ims = ims.astype(np.float32) / 255
    
    return ims


def get_balanced_floc_set(rndseed = 395878):

    np.random.seed(rndseed)
    
    min_per_set = n_each_food_categ
    
    csv_filename =  os.path.join(floc_image_root,'floc_image_labels.csv')
    df = pd.read_csv(csv_filename)
    n_images = df.shape[0]
    balanced_inds = np.zeros((n_images,),dtype=bool)

    for cc, categ in enumerate(categories):

        inds_all = np.where(df['category']==categ)[0]
        n_this_categ = len(inds_all)

        if n_this_categ>min_per_set:
            # choose a random n images
            inds_use = np.random.choice(inds_all, min_per_set, replace=False)
        else:
            # use all images
            inds_use = inds_all

        balanced_inds[inds_use] = True

    balanced_labels = np.array(df['category'])[balanced_inds]
    labs, counts = np.unique(balanced_labels, return_counts=True)
    assert(np.all(counts==min_per_set))
    
    save_filename =  os.path.join(floc_image_root,'balanced_floc_inds.npy')
    print('saving to %s'%save_filename)
    np.save(save_filename, {'image_inds_balanced': balanced_inds, 
                           'rndseed': rndseed})
    
    
def load_balanced_floc_set():
    
    filename =  os.path.join(floc_image_root,'balanced_floc_inds.npy')
    bal = np.load(filename, allow_pickle=True).item()
    
    return bal['image_inds_balanced']
