import numpy as np


def von_mises_deg(xx,mu,k,a=None,b=None,normalize=True,axis_size_deg = 180):
  
    """
    Make a von mises function over the range in xx.
    Input should be in 180 deg or 360 deg space, can specify space in degrees.
    mu = center
    k = concentration parameter
    a = amplitude (height)
    b = baseline
    if normalize=True, a and b are applied following normalization to range 0-1. 
    generally want to normalize first, otherwise height will vary with k.
    
    """ 
    if k<10**(-15):
        print('WARNING: k is too small, might get precision errors')
    
    xx_rad2pi = np.float128(xx/axis_size_deg*2*np.pi)
    mu_rad2pi = mu/axis_size_deg*2*np.pi
    yy = np.exp(k*(np.cos(xx_rad2pi-mu_rad2pi)-1))
    
    if normalize:
        # make the y values span from 0-1
        yy = yy-np.min(yy)
        yy = yy/np.max(yy)
    
    # then apply the entered amplitude and baseline.
    if a is not None:
        yy *= a
    if b is not None:
        yy += b

    return yy


def circ_corr_coef(x, y):
    """ calculate correlation coefficient between two circular variables
    Using Fisher & Lee circular correlation formula (code from Ed Vul)
    x, y are both in radians [0,2pi]
    
    """
   
    assert type(x)==np.ndarray
    assert type(y)==np.ndarray
    assert np.shape(x)==np.shape(y)
    if np.all(x==0) or np.all(y==0):
        raise ValueError('x and y cannot be empty or have all zero values')
    if np.any(x<0) or np.any(x>2*np.pi) or np.any(y<0) or np.any(y>2*np.pi):
        raise ValueError('x and y values must be between 0-2pi')
    n = np.size(x);
    assert(np.size(y)==n)
    A = np.sum(np.cos(x)*np.cos(y));
    B = np.sum(np.sin(x)*np.sin(y));
    C = np.sum(np.cos(x)*np.sin(y));
    D = np.sum(np.sin(x)*np.cos(y));
    E = np.sum(np.cos(2*x));
    Fl = np.sum(np.sin(2*x));
    G = np.sum(np.cos(2*y));
    H = np.sum(np.sin(2*y));
    corr_coef = 4*(A*B-C*D) / np.sqrt((np.power(n,2) - np.power(E,2) - np.power(Fl,2))*(np.power(n,2) - np.power(G,2) - np.power(H,2)));
   
    return corr_coef


def get_circ_peaks(curves):
    
    if len(curves.shape)==1:
        curves = curves[np.newaxis,:]
    circ_diffs = np.diff(np.concatenate([curves, curves[:,0:1]], axis=1), axis=1)
    circ_diffs_shifted = np.roll(circ_diffs, shift=1, axis=1)
    peaks = [ np.where((circ_diffs[ii,:]<0) & (circ_diffs_shifted[ii,:]>0))[0] for ii in range(curves.shape[0])]
    
    return peaks

def get_circ_troughs(curve):
    
    return get_circ_peaks(curve*(-1))