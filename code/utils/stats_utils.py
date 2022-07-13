import numpy as np
import scipy.stats
import warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import statsmodels.stats.multitest

def get_shared_unique_var(combined, just_a, just_b, \
                          remove_bad_voxels = False, \
                          convert_to_prop=False, \
                          enforce_prop_range=False):
    
    """
    Function for computing unique/shared variance based on R2 values for 
    full and partial models. 
    Input [R2 combined, R2 A solo, R2 B solo]
    Returns [shared variance, unique A, unique B]
    """
    
    unique_a = combined - just_b
    unique_b = combined - just_a
    shared_ab = just_a + just_b - combined
    
    vals = np.array([shared_ab, unique_a, unique_b]).T
    
    if remove_bad_voxels:
        # Sometimes this analysis results in negative values, or values that exceed the maximum variance
        # of the combined model. 
        # Can choose here to simply ignore voxels that have bad result, return NaNs.
        bad_inds = np.any(vals<0, axis=1) | np.any(vals>combined[:,None], axis=1)
        vals[bad_inds,:] = np.nan
        
    if convert_to_prop:
        # optionally convert the R2_shared and R2_unique values into a proportion
        # of combined model R2.
        vals /= np.tile(combined[:,None], [1,3])
        if enforce_prop_range:
            # force all the proportions to lie between 0 and 1.
            # note that this can make the sum over proportions not exactly=1
            # but it prevents negative var expl values.
            vals = np.maximum(np.minimum(vals, 1), 0)

    return vals[:,0], vals[:,1], vals[:,2]

def get_r2(actual,predicted):
    """
    This computes the coefficient of determination (R2).
    Always goes along first dimension (i.e. the trials/samples dimension)
    MAKE SURE INPUTS ARE ACTUAL AND THEN PREDICTED, NOT FLIPPED
    """
    ssres = np.sum(np.power((predicted - actual),2), axis=0);
    sstot = np.sum(np.power((actual - np.mean(actual, axis=0)),2), axis=0);
    r2 = 1-(ssres/sstot)
    
    return r2

def get_corrcoef(actual,predicted,dtype=np.float32):
    """
    This computes the linear correlation coefficient.
    Always goes along first dimension (i.e. the trials/samples dimension)
    Assume input is 2D.
    """
    assert(len(actual.shape)==2)
    vals_cc = np.full(fill_value=0, shape=(actual.shape[1],), dtype=dtype)
    for vv in range(actual.shape[1]):
        vals_cc[vv] = numpy_corrcoef_warn(actual[:,vv], predicted[:,vv])[0,1] 
    return vals_cc


def compute_partial_corr(x, y, c, return_p=False):

    """
    Compute the partial correlation coefficient between x and y, 
    controlling for the variables in covariates "c". 
    Uses linear regression based method.
    Inputs: 
        x [n_samples,] or [n_samples,1]
        y [n_samples,] or [n_samples,1]
        c [n_samples,] or [n_samples,n_covariates]
        
    Outputs:
        partial_corr, a single value for the partial correlation coefficient.
    """
    
    if len(x.shape)==1:
        x = x[:,np.newaxis]        
    if len(y.shape)==1:
        y = y[:,np.newaxis]
    if len(c.shape)==1:
        c = c[:,np.newaxis]
    n_trials = x.shape[0]
    assert(y.shape[0]==n_trials and c.shape[0]==n_trials)
    
    # first predict x from the other vars
    model1_preds = np.concatenate([c, np.ones((n_trials,1))], axis=1)
    model1_coeffs = np.linalg.pinv(model1_preds) @ x
    model1_yhat = model1_preds @ model1_coeffs
    model1_resids = model1_yhat - x
   
    # then predict y from the other vars
    model2_preds = np.concatenate([c, np.ones((n_trials,1))], axis=1)
    model2_coeffs = np.linalg.pinv(model2_preds) @ y
    model2_yhat = model2_preds @ model2_coeffs
    model2_resids = model2_yhat - y

    # correlate the residuals to get partial correlation.
    if return_p:
        partial_corr, p = scipy.stats.pearsonr(model1_resids[:,0], model2_resids[:,0])
        return partial_corr, p
    else:
        partial_corr = numpy_corrcoef_warn(model1_resids[:,0], model2_resids[:,0])[0,1]
        return partial_corr
   
    

def compute_partial_corr_formula(x,y,c):
 
    """
    Code to compute the partial correlation between x and y, controlling
    for covariate c. 
    Based on the correlation coefficients between each pair of variables.
    Also computes estimated standardized beta weight. 
    """
    x = np.squeeze(x); 
    y = np.squeeze(y);
    c = np.squeeze(c);
    ryx = np.corrcoef(x,y)[0,1]
    ryc = np.corrcoef(c,y)[0,1]
    rxc = np.corrcoef(x,c)[0,1]

    # partial correlation coefficient
    partial_corr = (ryx - ryc*rxc)/np.sqrt((1-ryc**2)*(1-rxc**2))
  
    # equivalent to standardized beta weight from a multiple linear regression
    # would be set up like [x, c, intercept] @ w = y
    # this is the weight for x.
    beta = (ryx - ryc*rxc)/(1-rxc**2)
    
    return partial_corr, beta


# Some functions that wrap basic numpy/scipy functions, but will print 
# more useful warnings when a problem arises

def numpy_corrcoef_warn(a,b):
    
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            cc = np.corrcoef(a,b)
        except RuntimeWarning as e:
            print('Warning: problem computing correlation coefficient')
            print('shape a: ',a.shape)
            print('shape b: ',b.shape)
            print('sum a: %.9f'%np.sum(a))
            print('sum b: %.9f'%np.sum(b))
            print('std a: %.9f'%np.std(a))
            print('std b: %.9f'%np.std(b))
            print(e)
            warnings.filterwarnings('ignore')
            cc = np.corrcoef(a,b)
            
    if np.any(np.isnan(cc)):
        print('There are nans in correlation coefficient')
    
    return cc


def ttest_warn(a,b):
    
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            ttest_out = scipy.stats.ttest_ind(a,b)
        except RuntimeWarning as e:
            print('Warning: problem with t test. Means/vars/counts each group:')
            groups = [a,b]
            means = [np.mean(group) for group in groups]
            vrs = [np.var(group) for group in groups]
            counts = [len(group) for group in groups]
            print(means)
            print(vrs)
            print(counts)
            print(e)
            warnings.filterwarnings('ignore')
            ttest_out = scipy.stats.ttest_ind(a,b)
    
    if np.any(np.isnan(ttest_out.statistic)):
        print('nans in t-test result')
           
    return ttest_out

def anova_oneway_warn(groups):
    
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            anova_out = scipy.stats.f_oneway(*groups)
        except RuntimeWarning as e:
            print('Warning: problem with one way anova. Means/vars/counts each group:')
            means = [np.mean(group) for group in groups]
            vrs = [np.var(group) for group in groups]
            counts = [len(group) for group in groups]
            print(means)
            print(vrs)
            print(counts)
            print(e)
            warnings.filterwarnings('ignore')
            anova_out = scipy.stats.f_oneway(*groups)
    
    if np.any(np.isnan(anova_out.statistic)):
        print('nans in anova result')
           
    return anova_out

def ttest_unequal(a,b):
    
    """
    T-test for unequal variances.
    Should behave like scipy.stats.ttest_ind, for equal_variance=False
    """
    assert((len(a.shape)==1) and (len(b.shape)==1))
    n1=len(a); n2=len(b);    
    
    # first compute sample variance for each group 
    # Bessel's correction; denominator = n-1
    sv1 = np.var(a)*n1/(n1-1)
    sv2 = np.var(b)*n2/(n2-1)
    
    denom = np.sqrt((sv1/n1 + sv2/n2))

    tstat = (np.mean(a) - np.mean(b))/denom
    
    return tstat

def ttest_equal(a,b):
    
    """
    T-test for equal variances.
    Should behave like scipy.stats.ttest_ind, for equal_variance=True
    """
    assert((len(a.shape)==1) and (len(b.shape)==1))
    n1=len(a); n2=len(b);   
    
    # first compute sample variance for each group 
    # Bessel's correction; denominator = n-1
    sv1 = np.var(a)*n1/(n1-1)
    sv2 = np.var(b)*n2/(n2-1)
    
    # Compute pooled sample variance
    pooled_var = ((n1-1)*sv1 + (n2-1)*sv2) / (n1+n2-2)
    denom = np.sqrt(pooled_var) * np.sqrt(1/n1+1/n2)

    tstat = (np.mean(a) - np.mean(b))/denom
   
    return tstat



def get_dprime(predlabs,reallabs,un=None):
    """ 
    Calculate d' for predicted and actual values. Works for multiple classes.
    """

    predlabs==np.squeeze(predlabs)
    reallabs==np.squeeze(reallabs)
    if len(predlabs)!=len(reallabs):
        raise ValueError('real and predicted labels do not match')
    if len(predlabs.shape)>1 or len(reallabs.shape)>1:
        raise ValueError('need to have 1d inputs')
    if un is None:
        un = np.unique(reallabs)
    if not np.all(np.isin(np.unique(predlabs), un)):
        print('Warning: some labels in pred are not included in real labels! Will return nan')
        return np.nan
    
    hrz=np.zeros((len(un),1));
    fpz=np.zeros((len(un),1));

    n_trials = len(predlabs);

    #loop over class labels, get a hit rate and false pos for each (treating
    #any other category as non-hit)
    for ii in range(len(un)):

        if np.sum(reallabs==un[ii])==0 or np.sum(reallabs!=un[ii])==0:

            # if one of the categories is completely absent - this will return a
            # nan dprime value
            return np.nan

        else:

            hr = np.sum((predlabs==un[ii]) & (reallabs==un[ii]))/np.sum(reallabs==un[ii]);
            fp = np.sum((predlabs==un[ii]) & (reallabs!=un[ii]))/np.sum(reallabs!=un[ii]);    

            # make sure this never ends up infinite
            # correction from Macmillan & Creelman, use 1-1/2N or 1/2N in place
            # of 1 or 0 
            if hr==0:
                hr=1/(2*n_trials)
            if fp==0:
                fp=1/(2*n_trials)
            if hr==1:
                hr=1-1/(2*n_trials)
            if fp==1:
                fp=1-1/(2*n_trials);

        # convert to z score (this is like percentile - so 50% hr would be zscore=0)
        hrz[ii]=scipy.stats.norm.ppf(hr,0,1);
        fpz[ii]=scipy.stats.norm.ppf(fp,0,1);

    # dprime is the mean of individual dprimes (for two classes, they will be
    # same value)
    dprime = np.mean(hrz-fpz);

    return dprime

def lin_reg(x,y):
   
    if len(x.shape)==1:
        x_mat = x[:,np.newaxis]
    if len(y.shape)==1:
        y_mat = y[:,np.newaxis]
        
    n_pts = x_mat.shape[0]
    assert(y_mat.shape[0]==n_pts)
    
    X = np.concatenate([x_mat, np.ones((n_pts,1))], axis=1)
    reg_coeffs = np.linalg.pinv(X) @ y_mat
    yhat = X @ reg_coeffs
    
    actual = np.squeeze(y_mat)
    pred = np.squeeze(yhat)
    ssres = np.sum(np.power((actual - pred),2));
    sstot = np.sum(np.power((actual - np.mean(actual)),2));
    r2 = 1-(ssres/sstot)
    
    return yhat, reg_coeffs, r2




def paired_ttest_nonpar(vals1, vals2, n_iter=1000, rndseed=None):
    
    if rndseed is None:
        rndseed = int(time.strftime('%M%H%d', time.localtime()))
    np.random.seed(rndseed)
        
    real_diff = np.mean(vals1-vals2)    
    
    shuff_diffs = np.zeros((n_iter,))
    
    for ii in range(n_iter):
        
        shuff_vals = np.array([vals1, vals2])
        # randomly swap the positions of values within a pair, with 50% prob
        which_swap = np.random.normal(0,1,[len(vals1),])>0
        shuff_vals[:,which_swap] = np.flipud(shuff_vals[:, which_swap])
    
        shuff_diffs[ii] = np.mean(shuff_vals[0,:]-shuff_vals[1,:])
    
    # pvalue for two-tailed test
    pval_twotailed = np.minimum( np.mean(shuff_diffs<=real_diff), \
                                 np.mean(shuff_diffs>=real_diff)) * 2
    
    return pval_twotailed, real_diff

def fdr_keepshape(pvals, alpha=0.05, method='indep'):
    
    """
    This is a wrapper for the fdr function in statsmodels, allows
    for entering a 2D array and FDR correct all values together.
    Returns arrays same shape as original.
    """
    orig_shape = pvals.shape
    pvals_reshaped = pvals.ravel()
    
    pvals_fdr, masked_fdr = statsmodels.stats.multitest.fdrcorrection(pvals_reshaped, alpha=alpha, method=method)
    
    pvals_fdr = np.reshape(pvals_fdr, orig_shape)
    masked_fdr = np.reshape(masked_fdr, orig_shape)
    
    return pvals_fdr, masked_fdr

def fdr(pvals, alpha=None, parametric=True):
    
    """
    % fdr() - compute false detection rate mask
    %
    % Usage:
    %   >> [p_fdr, p_masked] = fdr( pvals, alpha);
    %
    % Inputs:
    %   pvals   - vector or array of p-values
    %   alpha   - threshold value (non-corrected). If no alpha is given
    %             each p-value is used as its own alpha and FDR corrected
    %             array is returned.
    %   parametric - use parametric FDR? If False, use non-parametric FDR. Default=True
                   - parametric = B-H method, non-parametric = B-Y method
    %
    % Outputs:
    %   p_fdr    - pvalue used for threshold (based on independence
    %              or positive dependence of measurements)
    %   p_masked - p-value thresholded. Same size as pvals.
    %
    % Author: Arnaud Delorme, SCCN, 2008-
    %         Based on a function by Tom Nichols
    %
    % Reference: Bejamini & Yekutieli (2001) The Annals of Statistics

    % Copyright (C) 2002 Arnaud Delorme, Salk Institute, arno@salk.edu

    Ported from Matlab to Python (and modified slightly) by MMH, 2022
    
    """

    q = alpha

    p = np.sort(pvals.ravel())

    V = len(p);

    I = np.arange(1,V+1);

    if alpha is None:

        p_fdr = np.ones(pvals.shape)
        thresholds = np.exp(np.linspace(np.log(0.1),np.log(0.000001), 100));

        for thresh in thresholds:
            # calling the function recursively here
            _, p_masked = fdr(pvals, thresh);
            p_fdr[p_masked] = thresh   

    else:
        if parametric:
            # standard FDR 
            # B-H procedure, for positively correlated or independent tests
            c = 1;
        else:
            # non-parametric FDR 
            # B-Y procedure, for negatively correlated tests
            c = np.sum(1/(np.arange(1,V+1)))
        
        abv_thresh = np.where(p<=I/V*q/c)[0]

        if len(abv_thresh)>0:
            p_fdr = p[np.max(abv_thresh)]
        else:
            p_fdr = 0


    p_masked = pvals<=p_fdr
    
    return p_fdr, p_masked