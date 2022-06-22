import numpy as np
import os
from utils import default_paths

"""
General use functions for loading encoding model fit results.
Works with the fits generated and saved by code in model_fitting/fit_model.py
"""

def load_fit_results(subject, fitting_type, n_from_end=0, volume_space=True, \
                     verbose=True, root=None, return_filename=False):     
    """
    Load fit params for a given subject and fitting type. 
    By default, it loads the most recent fit (based on date str in name of saved file).
    If n_from_end is specified, will go back in time and load the nth most recent.
    """
    if root is None:
        root = default_paths.save_fits_path
    if volume_space:
        folder2load = os.path.join(root,'S%02d'%(subject), fitting_type)
    else:
        folder2load = os.path.join(root,'S%02d_surface'%(subject), fitting_type)
        
    # within this folder, assuming we want the most recent version that was saved
    files_in_dir = os.listdir(folder2load)
    from datetime import datetime
    my_dates = [f for f in files_in_dir if 'ipynb' not in f and 'DEBUG' not in f]
    try:
        my_dates.sort(key=lambda date: datetime.strptime(date, "%b-%d-%Y_%H%M_%S"))
    except:
        my_dates.sort(key=lambda date: datetime.strptime(date, "%b-%d-%Y_%H%M"))
    # if n from end is not zero, then going back further in time 
    most_recent_date = my_dates[-1-n_from_end]

    subfolder2load = os.path.join(folder2load, most_recent_date)
    file2load = os.path.join(subfolder2load, 'all_fit_params.npy')
   
    if verbose:
        print('loading from %s\n'%file2load)

    out = np.load(file2load, allow_pickle=True).item()
    
    if verbose:
        print(out.keys())
     
    if return_filename:
        return out, file2load
    else:
        return out
    
def print_output_summary(out):
    """
    Print all the keys in the saved data file and a summary of each value.
    """
    for kk in out.keys():
        if out[kk] is not None:
            if np.isscalar(out[kk]):
                print('%s = %s'%(kk, out[kk]))
            elif isinstance(out[kk],tuple) or isinstance(out[kk],list):
                print('%s: len %s'%(kk, len(out[kk])))
            elif isinstance(out[kk],np.ndarray):
                print('%s: shape %s'%(kk, out[kk].shape))
            elif isinstance:
                print('%s: unknown'%kk)
