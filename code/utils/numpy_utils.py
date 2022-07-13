import numpy as np
import scipy.io as sio
from scipy.special import erf
import math
import sys
import scipy.stats

def zscore_in_groups(data, group_labels):
    """
    Apply z-scoring to data of several columns at a time - column groupings given by group_labels.
    """
    if len(group_labels.shape)>1:
        group_labels = np.squeeze(group_labels)
    assert(len(group_labels.shape)==1)
    zdata = np.zeros(shape=np.shape(data))
    ungroups = np.unique(group_labels)
    for gg in range(len(ungroups)):       
        d = data[:,group_labels==ungroups[gg]]        
        zd = np.reshape(scipy.stats.zscore(d.ravel()), np.shape(d))        
        zdata[:,group_labels==ungroups[gg]] = zd
        
    return zdata

def zscore_in_groups_trntest(trndata, tstdata, group_labels):
    """
    Apply z-scoring to data of several columns at a time - column groupings given by group_labels.
    Computing mean/std on training set only, and apply same parameters to test set.
    """
    if len(group_labels.shape)>1:
        group_labels = np.squeeze(group_labels)
    assert(len(group_labels.shape)==1)
    trn_zdata = np.zeros(shape=np.shape(trndata))
    tst_zdata = np.zeros(shape=np.shape(tstdata))
    ungroups = np.unique(group_labels)
    for gg in range(len(ungroups)):       
        trnd = trndata[:,group_labels==ungroups[gg]] 
        tstd = tstdata[:,group_labels==ungroups[gg]] 
        trnmean = np.mean(trnd.ravel())
        trnstd = np.std(trnd.ravel())
        trn_zdata[:,group_labels==ungroups[gg]] = (trnd-trnmean)/trnstd
        tst_zdata[:,group_labels==ungroups[gg]] = (tstd-trnmean)/trnstd
        
    return trn_zdata, tst_zdata

def unshuffle(shuffled_data, shuffle_order):
    """
    Take an array that has been shuffled according to shuffle_order, and re-create its original order.
    Assumes that first dim of data is what needs to be unshuffled.    
    """   
    unshuffle_order = np.zeros_like(shuffle_order);
    unshuffle_order[shuffle_order] = np.arange(shuffled_data.shape[0])
    unshuffled_data = shuffled_data[unshuffle_order] # Unshuffle the shuffled data
    return unshuffled_data

def invertible_sort(sequence):
    """
    Sort a sequence and store the order needed to reverse sort.
    Based on np.argsort.
    """
    order2sort = np.argsort(sequence)
    order2reverse = np.argsort(order2sort)
    
    return order2sort, order2reverse

def double_sort(array, sort_by1, sort_by2):

    """
    Sort an array twice, first along one dimension and then by another dimension
    (i.e. within levels of the first dim)
    """
    if len(array)>1:
        array_orig = np.squeeze(array)
        assert(len(np.shape(array))==1)
        sort_by1 = np.squeeze(sort_by1)
        assert(len(np.shape(sort_by1))==1)
        sort_by2 = np.squeeze(sort_by2)
        assert(len(np.shape(sort_by2))==1)
    else:
        return array, [[0]]
    
    new_order = np.zeros(np.shape(array))
    vals1 = np.unique(sort_by1)
    start_ind=0
    for v1 in vals1:
        
        orig_inds = np.where(np.array(sort_by1)==v1)[0];
        
        new_inds = np.arange(start_ind, start_ind+len(orig_inds))
        start_ind+=len(orig_inds)
        
        sorder = np.argsort(np.array(sort_by2)[orig_inds])
        new_order[new_inds] = orig_inds[sorder]
        
    new_order = new_order.astype(int)
    
    return array[new_order], new_order

def iterate_range(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual 


def select_along_axis(a, choice, run_axis=0, choice_axis=1):
    ''' run axis of lenght N
        choice axis of lenght M
        choice is a vector of lenght N with integer entries between 0 and M (exclusive).
        Equivalent to:
        >   for i in range(N):
        >       r[...,i] = a[...,i,...,choice[i],...]
        returns an array with the same shape as 'a' minus the choice_axis dimension
    '''
    assert len(choice)==a.shape[run_axis], "underspecified choice"
    final_pos = run_axis - (1 if choice_axis<run_axis else 0)
    val = np.moveaxis(a, source=[run_axis, choice_axis], destination=[0,1])
    il = list(val.shape)
    il.pop(1)
    r = np.ndarray(shape=tuple(il), dtype=a.dtype)
    for i in range(len(choice)):
        r[i] = val[i,choice[i]]
    return np.moveaxis(r, source=0, destination=final_pos)

def get_list_size_gib(list_object):
    
    try:
        fullsize = np.sum([list_object[ii].nbytes for ii in range(len(list_object))])
    except:
        fullsize = np.sum([sys.getsizeof(list_object[ii]) for ii in range(len(list_object))])
    gib, gb = bytes_to_gb(fullsize)
    
    return gib
    
def bytes_to_gb(bytes_size):

    gib = bytes_size/1024**3
    gb = bytes_size/1000**3
    
    return gib, gb

def bin_ydata_by_xdata(xdata, ydata, n_bins, linear_bins=True, remove_nans=True, \
                       return_edges=False, return_std = False, use_unique=False):
           
    if len(xdata.shape)>1:
        xdata = np.squeeze(xdata)
    if len(ydata.shape)>1:
        ydata = np.squeeze(ydata)
    assert((len(xdata.shape)==1) and (len(ydata.shape)==1))
    
    if linear_bins:
        # break x axis into linearly spaced increments
        bin_edges = np.linspace(np.min(xdata), np.max(xdata)+0.0001, n_bins+1)
        bin_centers = bin_edges[0:-1]+(bin_edges[1]-bin_edges[0])/2
    elif use_unique:
        # force the bin centers to take on actual values in the data.
        bin_centers = np.unique(xdata)
        bin_edges = np.concatenate([bin_centers-0.01, [bin_centers[-1]+0.01]], axis=0)
        n_bins=len(bin_centers)
    else:
        # bin according to data density
        bin_edges = np.quantile(xdata, np.linspace(0,1,n_bins+1))
        bin_edges[-1]+=0.0001
        bin_centers = bin_edges[0:-1]+np.diff(bin_edges)/2

    xbinned = np.zeros((n_bins,))
    ybinned = np.zeros((n_bins,))
    ybin_std = np.zeros((n_bins,))
    used_yet = np.zeros((len(xdata),))
    for bb in range(n_bins):
        inds = (xdata>=bin_edges[bb]) & (xdata<bin_edges[bb+1])
        xbinned[bb] = bin_centers[bb]
        if np.sum(inds)>0:
            used_yet[inds] += 1
            ybinned[bb] = np.mean(ydata[inds])
            ybin_std[bb] = np.std(ydata[inds])
        else:
            ybinned[bb] = np.nan
            ybin_std[bb] = np.nan
            
    assert(np.all(used_yet)==1)
   
    if remove_nans:
        good = ~np.isnan(ybinned)
        xbinned=xbinned[good] 
        bin_edges=np.concatenate([bin_edges[0:-1][good], [bin_edges[-1]]], axis=0)
        ybinned=ybinned[good]
        ybin_std=ybin_std[good]
        
    to_return = xbinned, ybinned
    
    if return_edges:
        to_return += bin_edges,
    
    if return_std:
        to_return += ybin_std,
        
    return to_return


def list_all_combs(values, n_levels):
    
    values = np.squeeze(values)
    
    for ll in range(n_levels):
        if ll==0:
            array = values[:,None]
        else:
            array = np.concatenate([np.repeat(array, len(values), 0), \
                                np.tile(values[:,None], [array.shape[0],1])], axis=1)

    return array