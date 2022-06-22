
import os
import numpy as np
import h5py
from scipy.io import loadmat
import PIL.Image
import nibabel as nib
import pandas as pd
import pickle
import time

from utils import default_paths, roi_utils

nsd_root = default_paths.nsd_root;
stim_root = default_paths.stim_root
beta_root = default_paths.beta_root

trials_per_sess = 750
sess_per_subj = 40
# hard coded values based on sessions that are missing for some subs
max_sess_each_subj = [40,40,32,30,40,32,40,30]

def get_session_inds_full():
    session_inds = np.repeat(np.arange(0,sess_per_subj), trials_per_sess)
    return session_inds

def load_from_nii(nii_file):
    return nib.load(nii_file).get_fdata()
  
def load_from_mgz(mgz_file):
    return load_from_nii(mgz_file)
  
def load_from_hdf5(hdf5_file, keyname=None):
    data_set = h5py.File(hdf5_file, 'r')
    if keyname is None:
        keyname = list(data_set.keys())[0]
    values = np.copy(data_set[keyname])
    data_set.close()    
    return values

def image_uncolorize_fn(image):
    data = image.astype(np.float32) / 255
    if data.shape[2]==3:
        bw = (0.2126*data[:,:,0:1]+ 0.7152*data[:,:,1:2]+ 0.0722*data[:,:,2:3])
    elif data.shape[1]==3:
        bw = (0.2126*data[:,0:1]+ 0.7152*data[:,1:2]+ 0.0722*data[:,2:3])
    
    return bw

def image_preproc_fn(image):
    data = image.astype(np.float32) / 255
    return data

def ncsnr_to_nc(ncsnr, average_image_reps=False, subject=None):    
    """
    From Allen (2021) nature neuroscience.    
    """   
    if not average_image_reps: 
        # single trial data
        n = 1
        noise_ceiling = 100 * ncsnr**2 / (ncsnr**2 + 1/n)  
    else:
        if subject is None:
            # assume averaging over three reps of each image
            n = 3
            noise_ceiling = 100 * ncsnr**2 / (ncsnr**2 + 1/n)
        else:
            # for this subject, count how many reps of each image there actually were
            # assume all available sessions are being used
            image_order = get_master_image_order()    
            session_inds = get_session_inds_full()
            sessions = np.arange(max_sess_each_subj[subject-1])
            inds2use = np.isin(session_inds, sessions)
            image_order = image_order[inds2use]
            unique, counts = np.unique(image_order, return_counts=True)
            A = np.sum(counts==3)
            B = np.sum(counts==2)
            C = np.sum(counts==1)
            # special version of the NC formula, from nsd data manual
            noise_ceiling = 100 * ncsnr**2 / (ncsnr**2 + (A/3 + B/2 + C/1) / (A+B+C) )
            
    return noise_ceiling
    
def get_image_data(subject, random_images=False, native=False):

    """
    Load the set of NSD images that were shown to a given subject.
    This loads a subject-specific array of images, see [get_subject_specific_images] for details.
    Can also choose to insert random noise instead of the images here.
    """
    
    if random_images==False:        
        print('\nLoading images for subject %d\n'%subject)
        if native:
            image_data = load_from_hdf5(os.path.join(stim_root, 'S%d_stimuli_native.h5py'%subject))     
        else:
            image_data = load_from_hdf5(os.path.join(stim_root, 'S%d_stimuli_240.h5py'%subject))        
    else:        
        print('\nGenerating random gaussian noise images...\n')
        n_images = 10000
        image_data = (np.random.normal(0,1,[n_images, 3, 240,240])*30+255/2).astype(np.uint8)
        image_data = np.maximum(np.minimum(image_data, 255),0)

    print ('image data size:', image_data.shape, ', dtype:', image_data.dtype, ', value range:',\
        np.min(image_data[0]), np.max(image_data[0]))

    return image_data

def get_master_image_order():    
    """
    Gather the "ordering" information for NSD images.
    masterordering gives zero-indexed ordering of indices (matlab-like to python-like), same for all subjects. 
    consists of 30000 values in the range [0-9999], which provide a list of trials in order. 
    The value in ordering[ii] tells the index into the subject-specific stimulus array that we would need to take to
    get the image for that trial.
    """
    exp_design_file = os.path.join(nsd_root, 'nsddata','experiments','nsd','nsd_expdesign.mat')
    exp_design = loadmat(exp_design_file)
    
    image_order = exp_design['masterordering'].flatten() - 1 
    
    return image_order
      

def load_betas(subject, sessions=[0], voxel_mask=None,  zscore_betas_within_sess=True, volume_space=True):

    """
    Load preprocessed voxel data for an NSD subject (beta weights).
    Always loading the betas with suffix 'fithrf_GLMdenoise_RR.
    Concatenate the values across multiple sessions.
    "sessions" is zero-indexed, add one to get the actual session numbers.
    """
    
    if volume_space:
        beta_subj_folder = os.path.join(beta_root, 'subj%02d'%subject, 'func1pt8mm', 'betas_fithrf_GLMdenoise_RR')   
    else:
        beta_subj_folder = os.path.join(beta_root, 'subj%02d'%subject, 'nativesurface', 'betas_fithrf_GLMdenoise_RR')   

    print('Data is located in: %s...'%beta_subj_folder)

    if np.any((np.array(sessions)+1)>max_sess_each_subj[subject-1]):
        print('attempting to load sessions:')
        print(sessions+1)
        raise ValueError('trying to load sessions that do not exist for subject %d, only has up to session %d'\
                         %(subject, max_sess_each_subj[subject-1]))
        
    n_trials = len(sessions)*trials_per_sess

    for ss, se in enumerate(sessions):

        if volume_space:

            # Load volume space nifti
            fn2load = os.path.join(beta_subj_folder, 'betas_session%02d.nii.gz'%(se+1))
            print('Loading from %s...'%fn2load)
            values = load_from_nii(fn2load).transpose((3,0,1,2))
            print('Raw data:')
            print(values.dtype, np.min(values), np.max(values), values.shape)

            betas = values.reshape((len(values), -1), order='C')

        else:
            # Surface space, concatenate the two hemispheres
            # Must be left then right to match ROI definitions.
            fn2load1 = os.path.join(beta_subj_folder, 'lh.betas_session%02d.hdf5'%(se+1))
            fn2load2 = os.path.join(beta_subj_folder, 'rh.betas_session%02d.hdf5'%(se+1))

            print('Loading from %s...'%fn2load1)        
            values1 = load_from_hdf5(fn2load1)
            print('Raw data:')
            print(values1.dtype, np.min(values1), np.max(values1), values1.shape)

            print('Loading from %s...'%fn2load2)        
            values2 = load_from_hdf5(fn2load2)
            print('Raw data:')
            print(values2.dtype, np.min(values2), np.max(values2), values2.shape)

            betas = np.concatenate((values1, values2), axis=1)

        # divide by 300 to convert back to percent signal change
        betas = betas.astype(np.float32) / 300

        print('Adjusted data (divided by 300):')
        print(betas.dtype, np.min(betas), np.max(betas), betas.shape)
        
        if voxel_mask is not None:        
            betas = betas[:,voxel_mask]

        if zscore_betas_within_sess: 
            print('z-scoring beta weights within this session...')
            mb = np.mean(betas, axis=0, keepdims=True)
            sb = np.std(betas, axis=0, keepdims=True)
            betas = np.nan_to_num((betas - mb) / (sb + 1e-6))
            print ("mean = %.3f, sigma = %.3f" % (np.mean(mb), np.mean(sb)))

        if ss==0:        
            n_vox = betas.shape[1]
            betas_full = np.zeros((n_trials, n_vox))   

        betas_full[ss*trials_per_sess : (ss+1)*trials_per_sess, :] = betas
        
    return betas_full
       
def get_concat_betas(subject, debug=False):
    
    print('\nProcessing subject %d\n'%subject)
    voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = \
                roi_utils.get_voxel_roi_info(subject,volume_space=True)    
    if debug:
        sessions = [0]
    else:
        sessions = np.arange(max_sess_each_subj[subject-1])
    zscore_betas_within_sess = True
    volume_space=True
    voxel_data = load_betas(subject, sessions, voxel_mask=voxel_mask, \
                                      zscore_betas_within_sess=zscore_betas_within_sess, \
                                      volume_space=volume_space)
    print('\nSize of full data set [nTrials x nVoxels] is:')
    print(voxel_data.shape)

    save_fn = os.path.join(default_paths.nsd_data_concat_root, 'S%d_allsess_concat_visual.h5py'%subject)

    t = time.time()
    print('saving file to %s'%save_fn)
    with h5py.File(save_fn, 'w') as data_set:
        dset = data_set.create_dataset("betas", np.shape(voxel_data), dtype=np.float64)
        data_set['/betas'][:,:] = voxel_data
        data_set.close()  
    elapsed = time.time() - t
    print('took %.5f sec'%elapsed)
    
def average_image_repetitions(voxel_data, image_order):
    
    n_trials = voxel_data.shape[0]
    n_voxels = voxel_data.shape[1]

    unique_ims = np.unique(image_order)
    n_unique_ims = len(unique_ims)
    avg_dat_each_image = np.zeros((n_unique_ims, n_voxels))
    for uu, im in enumerate(unique_ims):
        inds = image_order==im;
        avg_dat_each_image[uu,:] = np.mean(voxel_data[inds,:], axis=0)
        
    return avg_dat_each_image, unique_ims
    
def get_data_splits(subject, sessions=[0], voxel_mask=None, \
                    zscore_betas_within_sess=True, volume_space=True, \
                    average_image_reps = False, \
                    shuffle_images=False, random_voxel_data=False):

    """
    Gather voxel data and the indices of training/testing images, for one NSD subject.
    Not actually loading images here, because all image features are pre-computed. 
    Always leaving out the "shared1000" image subset as my validation set, and training within the rest of the data.
    Can specify a list of sessions to work with (don't have to be contiguous).
    Can specify whether to work in volume or surface space (set volume_space to True or False).
    Can also choose to shuffle images or generate random voxel data at this stage if desired.
    """

    # Load the experiment design file that defines full image order over 30,000 trials
    image_order = get_master_image_order()
    
    # Decide which sessions to work with here
    session_inds = get_session_inds_full()
    if np.isscalar(sessions):
        sessions = [sessions]
    sessions = np.array(sessions)    
    if np.any((sessions+1)>max_sess_each_subj[subject-1]):
        # adjust the session list that was entered, if the subject is missing some sessions.
        # will alter the list for both images and voxel data.
        print('subject %d only has up to session %d, will load these sessions:'%\
              (subject, max_sess_each_subj[subject-1]))
        sessions = sessions[(sessions+1)<=max_sess_each_subj[subject-1]]
        print(sessions+1)
        assert(len(sessions)>0)
        
    inds2use = np.isin(session_inds, sessions)
    session_inds = session_inds[inds2use]
    image_order = image_order[inds2use]

    # Now load voxel data (preprocessed beta weights for each trial)
    print('Loading data for sessions:')
    print(sessions+1)
    if not random_voxel_data:
        voxel_data = load_betas(subject, sessions, voxel_mask=voxel_mask, \
                            zscore_betas_within_sess=zscore_betas_within_sess, \
                            volume_space=volume_space)
    else:
        print('creating random normally distributed data instead of loading real data')
        if voxel_mask is not None:
            n_voxels = np.sum(voxel_mask)
        else:
            n_voxels = 10000
        voxel_data = np.random.normal(0,1,(len(image_order), n_voxels))
        
    print('\nSize of full data set [n_trials x n_voxels] is:')
    print(voxel_data.shape)
    assert(voxel_data.shape[0]==len(image_order))
    
    # average over repetitions of same image, if desired
    if average_image_reps:
        avg_dat_each_image, unique_ims = average_image_repetitions(voxel_data, image_order)
        voxel_data = avg_dat_each_image # use average data going forward
        image_order = unique_ims # now the unique image indices become new image order
        # NOTE that the unique images can be fewer than 10,000 if the subject
        # is missing some data, or if we are working w just a few sessions. 
        print('\nAfter averaging - size of full data set [n_images x n_voxels] is:')
        print(voxel_data.shape)
        # can't have session inds here because betas are averaged over multiple sessions
        session_inds=None
        
    if shuffle_images:
        print('\nShuffling image order')
        image_order = image_order[np.random.permutation(np.arange(len(image_order)))]
   
    # Get indices to split into training/validation set now
    subj_df = get_subj_df(subject)
    is_shared_image = np.array(subj_df['shared1000'])
    shared_1000_inds = is_shared_image[image_order]
    val_inds = shared_1000_inds
    
    is_trn, is_holdout, is_val = load_image_data_partitions(subject)
    is_val = is_val[image_order]
    is_holdout = is_holdout[image_order]
    assert(np.all(is_val==val_inds))
    holdout_inds = is_holdout

    return voxel_data, image_order, val_inds, holdout_inds, session_inds
   

def resize_image_tensor(x, newsize):
        tt = x.transpose((0,2,3,1))
        r  = np.ndarray(shape=x.shape[:1]+newsize+(x.shape[1],), dtype=tt.dtype) 
        for i,t in enumerate(tt):
            r[i] = np.asarray(PIL.Image.fromarray(t).resize(newsize, resample=PIL.Image.BILINEAR))
        return r.transpose((0,3,1,2))   

def get_subject_specific_images(nsd_root, path_to_save, npix=227):

    """ 
    Load the big array of NSD images for all subjects.
    Downsample to a desired size, and select just those viewed by a given subject.
    Save a smaller array for each subject, at specified path.
    """
    
    stim_file_original = os.path.join(nsd_root,"nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")
    exp_design_file = os.path.join(nsd_root,"nsddata/experiments/nsd/nsd_expdesign.mat")
    exp_design = loadmat(exp_design_file)
    subject_idx  = exp_design['subjectim']
    
    print ("Loading full block of images...")
    image_data_set = h5py.File(stim_file_original, 'r')
    print(image_data_set.keys())
    image_data = np.copy(image_data_set['imgBrick'])
    image_data_set.close()
    print(image_data.shape)

    for k,s_idx in enumerate(subject_idx):
        fn2save = os.path.join(path_to_save, 'S%d_stimuli_%d'%(k+1, npix))
        print('Will save to %s'%fn2save)       
        print('Resizing...')
        s_image_data = image_data[s_idx - 1]
        s_image_data = resize_image_tensor(s_image_data.transpose(0,3,1,2), newsize=(npix,npix))
        print(s_image_data.shape)        
        print('saving to %s'%fn2save)
        
        with h5py.File(fn2save + '.h5py', 'w') as hf:
            key='stimuli'
            val=s_image_data        
            hf.create_dataset(key,data=val)
            print ('saved %s in h5py file' %(key))


def get_subj_df(subject):
    """
    Get info about the 10,000 images that were shown to each subject.
    Note this is not the full ordered sequence of trials (which is 30,000 long)
    This is only the unique images 
    (matches what is in /user_data/mmhender/nsd_stimuli/stimuli/nsd/S1_stimuli....h5py)
    """
    exp_design_file = os.path.join(nsd_root,"nsddata/experiments/nsd/nsd_expdesign.mat")
    exp_design = loadmat(exp_design_file)
    subject_idx  = exp_design['subjectim']
    
    nsd_meta_file = os.path.join(nsd_root, 'nsddata/experiments/nsd/nsd_stim_info_merged.pkl')
    with open(nsd_meta_file,'rb') as f:
        stim_info = pickle.load(f,encoding="latin1")
    
    ss=subject-1
    subject_df = stim_info.loc[subject_idx[ss,:]-1]

    return subject_df


def load_prf_mapping_pars(subject, voxel_mask=None):
    
    """
    Load parameters of pRF fits for each voxel, obtained during independent pRF mapping expt.
    Stimuli are sweeping bars w objects, see:
    https://natural-scenes-dataset.s3-us-east-2.amazonaws.com/nsddata/experiments/prf/prf_screencapture.mp4
    """
    
    if voxel_mask is None:
        voxel_mask, _, _, _, _ = \
        roi_utils.get_voxel_roi_info(subject, volume_space=True, include_all=True, \
                           include_body=True,verbose=False)
        
    prf_path = os.path.join(default_paths.nsd_root, 'nsddata','ppdata','subj%02d'%subject,'func1pt8mm')

    angle = load_from_nii(os.path.join(prf_path, 'prf_angle.nii.gz')).flatten()[voxel_mask]
    eccen = load_from_nii(os.path.join(prf_path, 'prf_eccentricity.nii.gz')).flatten()[voxel_mask]
    size = load_from_nii(os.path.join(prf_path, 'prf_size.nii.gz')).flatten()[voxel_mask]
    exponent = load_from_nii(os.path.join(prf_path, 'prf_exponent.nii.gz')).flatten()[voxel_mask]
    gain = load_from_nii(os.path.join(prf_path, 'prf_gain.nii.gz')).flatten()[voxel_mask]
    rsq = load_from_nii(os.path.join(prf_path, 'prf_R2.nii.gz')).flatten()[voxel_mask]/100
            
    return angle, eccen, size, exponent, gain, rsq


def load_domain_tvals(subject, voxel_mask=None):

    """
    For one NSD subject, load the t-statistics for all domain contrasts 
    from independent localizer task (faces, places, etc)
    """
    
    if voxel_mask is None:
        voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = \
                roi_utils.get_voxel_roi_info(subject, volume_space=True)
    
    n_voxels = np.sum(voxel_mask)

    niftis_path = os.path.join(default_paths.nsd_root, \
                               'nsddata', 'ppdata','subj%02d'%subject, 'func1pt8mm')

    categ_list = ['places', 'faces', 'bodies', 'objects', 'characters']
    n_categ = len(categ_list)

    tvals_all = np.zeros((n_voxels, n_categ))

    for cc, categ in enumerate(categ_list):

        # load t-statistics for the domain contrast of interest
        tvals_filename = os.path.join(niftis_path, 'floc_%stval.nii.gz'%categ)
        tvals = load_from_nii(tvals_filename)
        tvals = tvals.reshape((1, -1), order='C')[0]

        # pull out same set of voxels all my analyses were done on
        tvals_masked = tvals[voxel_mask] 

        tvals_all[:,cc] = tvals_masked

    return tvals_all, categ_list


def get_image_ranks(subject, sessions=np.arange(0,40), debug=False):
    
    """
    For each voxel, rank images in order of average response 
    (averaged over duplicate trials) and save as a csv file.
    Each column in csv is a voxel, each row is a rank position.
    """

    if np.isscalar(sessions):
        sessions = [sessions]
    sessions = np.array(sessions)
    if np.any((sessions+1)>max_sess_each_subj[subject-1]):
        # adjust the session list that was entered, if the subject is missing some sessions.
        # will alter the list for both images and voxel data.
        print('subject %d only has up to session %d, will load these sessions:'%(subject, max_sess_each_subj[subject-1]))
        sessions = sessions[(sessions+1)<=max_sess_each_subj[subject-1]]
        print(sessions+1)
        assert(len(sessions)>0)
    if debug:
        sessions = np.array([0])
        
    voxel_mask, _, _, _, _ = roi_utils.get_voxel_roi_info(subject, \
                                                volume_space=True, include_all=True, \
                                                include_body=True, verbose=False)
    voxel_data = load_betas(subject, sessions, voxel_mask=voxel_mask, \
                              zscore_betas_within_sess=True, \
                              volume_space=True)    
    image_order = get_master_image_order()
    session_inds = get_session_inds_full()

    inds2use = np.isin(session_inds, sessions)
    image_order = image_order[inds2use]

    n_trials = voxel_data.shape[0]
    n_voxels = voxel_data.shape[1]

    unique_ims = np.unique(image_order)
    n_unique_ims = len(unique_ims)
    avg_dat_each_image = np.zeros((n_unique_ims, n_voxels))
    for uu, im in enumerate(unique_ims):
        if debug and (uu>1):
            continue
        inds = image_order==im;
        avg_dat_each_image[uu,:] = np.mean(voxel_data[inds,:], axis=0)


    images_ranked_each_voxel = np.zeros((n_unique_ims, n_voxels))
    for vv in range(n_voxels):
        if debug and (vv>1):
            continue
        image_rank = np.flip(np.argsort(avg_dat_each_image[:,vv]))
        images_ranked = unique_ims[image_rank]
        images_ranked_each_voxel[:,vv] = images_ranked

    rank_df = pd.DataFrame(data=images_ranked_each_voxel.astype(int), \
                           columns=['voxel %d'%vv for vv in range(n_voxels)])

    fn2save = os.path.join(default_paths.stim_root, 'S%d_ranked_images.csv'%subject)
    print('Saving to %s'%fn2save)
    rank_df.to_csv(fn2save, header=True)
    
def load_image_data_partitions(subject):
    
    fn2load = os.path.join(default_paths.stim_root, 'Image_data_partitions.npy')
    print('loading train/holdout/val image list from %s'%fn2load)
    partitions = np.load(fn2load, allow_pickle=True).item()
    is_trn = partitions['is_trn'][:,subject-1]
    is_holdout = partitions['is_holdout'][:,subject-1]
    is_val = partitions['is_val'][:,subject-1]
    
    return is_trn, is_holdout, is_val

def make_image_data_partitions(pct_holdout=0.10):

    subjects=np.concatenate([np.arange(1,9),[999]], axis=0)
    n_subjects = len(subjects)
    # fixed random seeds for each subject, to make sure shuffling is repeatable
    rndseeds = [171301, 42102, 490304, 521005, 11407, 501610, 552211, 450013, 824387]
    
    n_images_total = 10000
    is_trn = np.zeros((n_images_total,n_subjects),dtype=bool)
    is_holdout = np.zeros((n_images_total,n_subjects),dtype=bool)
    is_val = np.zeros((n_images_total,n_subjects),dtype=bool)

    for si, ss in enumerate(subjects):

        if ss==999:
            val_image_inds = np.arange(0,10000)<1000
            trn_image_inds = np.arange(0,10000)>=1000            
        else:           
            subject_df = get_subj_df(ss)
            val_image_inds = subject_df['shared1000']
            trn_image_inds = ~subject_df['shared1000']

        n_images_val = np.sum(val_image_inds)
        n_images_notval = np.sum(trn_image_inds);
        n_images_holdout = int(np.ceil(n_images_notval*pct_holdout))
        n_images_trn = n_images_notval - n_images_holdout

        # of the full 9000 image training set, holding out a random chunk
        inds_notval = np.where(trn_image_inds)[0]
        np.random.seed(rndseeds[si])
        np.random.shuffle(inds_notval)
        inds_trn = inds_notval[0:n_images_trn]
        inds_holdout = inds_notval[n_images_trn:]
        assert(len(inds_holdout)==n_images_holdout)

        is_trn[inds_trn,si] = 1
        is_holdout[inds_holdout,si] = 1
        is_val[val_image_inds,si] = 1

    fn2save = os.path.join(default_paths.stim_root, 'Image_data_partitions.npy')
    np.save(fn2save, {'is_trn': is_trn, \
                      'is_holdout': is_holdout, \
                      'is_val': is_val, \
                      'rndseeds': rndseeds})