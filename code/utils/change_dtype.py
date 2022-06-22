import sys, os
import numpy as np
import h5py
import time

from utils import default_paths

def change_h5py_dtype(h5py_fn, dtype=np.float32, fieldname='features'):
    
    t=time.time()
    with h5py.File(h5py_fn, 'r') as data_set:
        dtype_orig = data_set['/%s'%fieldname].dtype
        print('original type:')
        print(dtype_orig)
        shape_orig = data_set['/%s'%fieldname].shape
        print('original size:')
        print(shape_orig)
        data_orig = np.zeros(data_set['/%s'%fieldname].shape, dtype=np.float32)
        n_batches=10; 
        batch_size = int(np.ceil(shape_orig[0]/n_batches));
        for bb in range(n_batches):
            print('loading batch %d of %d'%(bb, n_batches))
            batch_inds = np.arange(batch_size*bb, np.min([batch_size*(bb+1),shape_orig[0]]))            
            data_orig[batch_inds,:,:] = np.array(data_set['/%s'%fieldname][batch_inds,:,:]).astype(np.float32)
        print('first element:')
        print(data_orig[0,0,0])
        data_set.close() 
    elapsed = time.time() - t;
    print('took %.5f sec to load file'%elapsed)
    
    if dtype_orig==dtype:
        print('file %s is already in desired data type (%s), exiting'%(h5py_fn,dtype))
    else:
        print('changing dtype of %s to %s'%(h5py_fn, dtype))
        t=time.time()
        with h5py.File(h5py_fn, 'w') as data_set:
            dset = data_set.create_dataset(fieldname, np.shape(data_orig), dtype=dtype)
            data_set['/features'][:,:,:] = data_orig
            data_set.close()
        elapsed = time.time() - t;
        print('took %.5f sec to write new file'%elapsed)
        
        t=time.time()
        with h5py.File(h5py_fn, 'r') as data_set:
            print('new size:')
            print(data_set['/%s'%fieldname].shape)
            print('first element:')
            print(data_set['/%s'%fieldname][0,0,0])
            print('new type:')
            print(data_set['/%s'%fieldname].dtype)
            data_set.close() 
        elapsed = time.time() - t;
        print('took %.5f sec to load file'%elapsed)
        
if __name__ == '__main__':
       
#     subs2do = [8]
    subs2do = np.arange(1,8,1)
    features_folder = os.path.join(default_paths.root, 'features', 'pyramid_texture')
    for subject in subs2do:
        h5py_fn = os.path.join(features_folder,'S%d_features_each_prf_4ori_4sf_grid5.h5py'%subject)
        change_h5py_dtype(h5py_fn, dtype=np.float32, fieldname='features')

    features_folder = os.path.join(default_paths.root, 'features', 'sketch_tokens')
    for subject in subs2do:
        h5py_fn = os.path.join(features_folder,'S%d_features_each_prf_grid5.h5py'%subject)
        change_h5py_dtype(h5py_fn, dtype=np.float32, fieldname='features')

    features_folder = os.path.join(default_paths.root, 'features', 'gabor_texture')
    for subject in subs2do:
        h5py_fn = os.path.join(features_folder,'S%d_features_each_prf_12ori_8sf_gabor_solo_nonlin_grid5.h5py'%subject)
        change_h5py_dtype(h5py_fn, dtype=np.float32, fieldname='features')
