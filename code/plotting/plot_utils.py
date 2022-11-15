import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import copy

def set_all_font_sizes(fs):
    
    plt.rc('font', size=fs)          # controls default text sizes
    plt.rc('axes', titlesize=fs)     # fontsize of the axes title
    plt.rc('axes', labelsize=fs)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fs)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fs)    # fontsize of the tick labels
    plt.rc('legend', fontsize=fs)    # legend fontsize
    plt.rc('figure', titlesize=fs)  # fontsize of the figure title

def create_roi_subplots(data, inds2use, single_plot_object, roi_def, \
                        subject_inds=None,skip_inds=None, suptitle=None, \
                        label_just_corner=True, figsize=None):

    """
    Create a grid of subplots for each ROI in our dataset.
    
    data: array [n_voxels x n_measures] where n_measures can be encoding model performance, 
        unique variance explained by a set of features, etc.
    inds2use: list or 1D array, a boolean mask for which of n_voxels to use 
        (based on for example R2 threshold)
    single_plot_object: either a scatter_plot, violin_plot, or bar_plot (see below)
    roi_def: roi definition object, from roi_utils.nsd_roi_def()
    subject_inds: optional, list or 1D array as long as n_voxels, indicates which subject
        the voxels are each from. If making a scatter plots, this will control what color 
        the points are each plotted. If making a bar plot, this will control how voxels are
        averaged (first within subject, then across). If making a violin plot, this does 
        nothing currently.
    skip_inds: which of the n_rois in roi_def do you want to skip?
    suptitle: optional string for a sup-title
    label_just_corner: boolean, want to add axis labels just to the lower left plot?
    figsize: optional figure size for entire plot
    
    """

    n_rois = roi_def.n_rois
    roi_names = roi_def.roi_names
   
    if skip_inds is None:
        skip_inds = []
        
    if subject_inds is not None:
        assert(len(subject_inds)==data.shape[0])
        un_subs = np.unique(subject_inds)
        n_subjects = len(un_subs)
    else:
        n_subjects = 1
        
    if hasattr(single_plot_object, 'plot_counts') and single_plot_object.plot_counts:
        plot_counts=True
        single_plot_object.plot_counts = False 
        if single_plot_object.groups is None:
            single_plot_object.groups = np.unique(data)  
    else:
        plot_counts=False
        
    if figsize==None:
        figsize=(24,20)
    plt.figure(figsize=figsize)
    npy = int(np.ceil(np.sqrt(n_rois-len(skip_inds))))
    npx = int(np.ceil((n_rois-len(skip_inds))/npy))

    if label_just_corner:
        pi2label = [(npx-1)*npy+1]
    else:
        pi2label = np.arange(npx*npy+1)
        
    pi = 0
    for rr in range(n_rois):

        if rr not in skip_inds:
            
            inds_this_roi = roi_def.get_indices(rr)

            data_this_roi = data[inds2use & inds_this_roi,:]
            if subject_inds is not None:
                subject_inds_this_roi = subject_inds[inds2use & inds_this_roi]
            else:   
                subject_inds_this_roi = None
                
            pi = pi+1
            plt.subplot(npx, npy, pi)

            single_plot_object.title = '%s (%d vox)'%(roi_names[rr], data_this_roi.shape[0])
            if pi in pi2label:
                minimal_labels=False
            else:
                minimal_labels=True
            if hasattr(single_plot_object, 'plot_errorbars'):
                # bar plot case
                if data_this_roi.shape[0]>0:
                    if plot_counts:  
                        # compute number of voxels in each group
                        counts_each_subj = np.array([[np.sum(data_this_roi==gg) \
                                                      for gg in single_plot_object.groups] \
                                                      for si in un_subs])
                        mean_this_roi = np.mean(counts_each_subj, axis=0)
                        if n_subjects>1:                           
                            err_this_roi = np.std(counts_each_subj, axis=0)/np.sqrt(n_subjects)
                        else:
                            err_this_roi = np.zeros(np.shape(mean_this_roi))
                    else:
                        # take mean and sem over all pts.
                        if n_subjects>1:
                            # >1 subject, average +/- SEM over subjects
                            mean_each_subj = np.array([np.mean(data_this_roi[subject_inds_this_roi==si,:], axis=0) \
                                              for si in un_subs])
                            mean_this_roi = np.mean(mean_each_subj, axis=0)
                            err_this_roi = np.std(mean_each_subj, axis=0)/np.sqrt(n_subjects)
                        else:
                            # one subject, average +/- SEM over voxels
                            mean_this_roi = np.mean(data_this_roi, axis=0)
                            err_this_roi = np.std(data_this_roi, axis=0)/np.sqrt(data_this_roi.shape[0])
                    single_plot_object.create(mean_this_roi, err_data=err_this_roi, new_fig=False, \
                                              minimal_labels=minimal_labels)
            else:
                # other plot types, passing in raw data.
                single_plot_object.create(data_this_roi, new_fig=False, minimal_labels=minimal_labels, \
                                      subject_inds=subject_inds_this_roi)

    if suptitle is not None:
        plt.suptitle(suptitle)
        
        
class bar_plot:
       
    """
    A bar plot object that plots the means of each column in a data matrix.
    
    Can be passed to "create_roi_subplots" to draw many barplots.
    
    colors: optional array of colors, [n_bars x 3]
    column_labels: optional list of strings, [n_bars]
    plot_errorbars: boolean, want to add SEM errorbars to your plot?
    ylabel: optional, string for yaxis label
    yticks: optional, list of y-axis ticks to use
    ylims: optional, tuple of y-lims
    title: optional title for the barplot
    horizontal_line_pos: optional, position to draw a horizontal line on the plot    
    plot_counts: boolean, are we going to plot a histogram? if yes, must specify "groups"
    groups: list of groups to do counts in, if plot_counts=True
    input_stats: is the data you will enter a summary statistic (mean and SEM)? else assume it 
        is raw data, and compute mean/SEM in this code.
    Use "create" method to pass in data and make the plot.
    data: [n_samples x n_bars], the bar heights will be the mean of each column. 
    new_fig: boolean, if True create a new figure, if False assume we already have a figure open.
    figsize: optional figure size
    minimal_labels: boolean, if True the plot will not have xticks/yticks/xlims/ylims
    
    """
    
    def __init__(self, colors=None, column_labels=None, plot_errorbars=True, \
                 ylabel=None, yticks=None, ylims=None, title=None, \
                 horizontal_line_pos=None, plot_counts=False, groups=None, \
                 input_stats=False):

        self.colors = colors
        self.column_labels = column_labels
        self.plot_errorbars=plot_errorbars
        self.plot_counts = plot_counts
        if self.plot_counts:
            self.plot_errorbars=False
        self.ylabel = ylabel
        self.title = title
        self.horizontal_line_pos = horizontal_line_pos
        self.ylims = ylims
        self.yticks = yticks      
        self.groups = groups
        self.input_stats = input_stats
        
    def create(self, data, err_data=None, new_fig=True, figsize=None, minimal_labels=False, **kwargs):
    
        data = np.squeeze(data)
        assert(len(data.shape)==1)
        if self.plot_errorbars:
            assert(err_data is not None)
            err_data = np.squeeze(err_data)
            assert(np.all(err_data.shape==data.shape))
        
        if new_fig:
            if figsize is None:
                figsize=(16,8)
            plt.figure(figsize=figsize)
  
        if self.plot_counts:
            if self.groups is None:
                self.groups = np.unique(data)                
            counts = np.array([np.sum(data==gg) for gg in self.groups])
            data = counts
            
        # n_columns=n_bars
        n_columns = len(data)
          
        if self.colors is None:
            colors = cm.plasma(np.linspace(0,1,n_columns))
            colors = np.flipud(colors)
        else:
            colors = self.colors

        for cc in range(n_columns):
            mean = data[cc]
            plt.bar(cc, mean, color=colors[cc,:])
            if self.plot_errorbars:
                err = err_data[cc]
                plt.errorbar(cc, mean, err, color = colors[cc,:], ecolor='k', zorder=10)

        if not minimal_labels:
            if self.column_labels is not None:
                plt.xticks(ticks=np.arange(0,n_columns),labels=self.column_labels,\
                           rotation=45, ha='right',rotation_mode='anchor')
            if self.ylabel is not None:
                plt.ylabel(self.ylabel)
            if self.yticks is not None:
                plt.yticks(self.yticks)
        else:
            plt.yticks([])
            plt.xticks([])
            
        if self.title is not None:
            plt.title(self.title)
        if self.horizontal_line_pos is not None:
            plt.axhline(self.horizontal_line_pos, color=[0.8, 0.8, 0.8])
        if self.ylims is not None:
            plt.ylim(self.ylims)
            
            
class scatter_plot:
       
    """
    A scatter plot object that plots the relationship between two columns of data.
    
    Can be passed to "create_roi_subplots" to draw many scatterplots.
    
    colors: optional color for the dots
    xlabel/ylabel: optional, string for x-axis/y-axis label
    xticks/yticks: optional, list of x-axis/y-axis ticks to use
    xlims/ylims: optional, tuple of x-lims/y-lims   
    title: optional title for the plot
    show_diagonal: boolean, do you want to add a diagonal line with slope 1/intercept 0?
    show_axes: boolean, do you want to draw horizontal and vertical lines at x=0, y=0?
    square: boolean, do you want to make the axes square?
    add_best_fit_lines: do you want to perform linear regression of y onto x and plot yhat?
    
    Use "create" method to pass in data and make the plot.
    data: [n_samples x 2], or [xvals, yvals] 
    subject_inds: [n_samples,], provides index into self.colors for each point.
    new_fig: boolean, if True create a new figure, if False assume we already have a figure open.
    figsize: optional figure size
    minimal_labels: boolean, if True the plot will not have xticks/yticks/xlims/ylims
    
    """
        
    def __init__(self, color=None, xlabel=None, ylabel=None, xlims=None, \
                 ylims=None, xticks=None, yticks=None, \
                 title=None, show_diagonal=True, show_axes=True, square=True, \
                 add_best_fit_lines=False):

        self.color = color
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlims = xlims
        self.ylims = ylims
        self.xticks = xticks
        self.yticks = yticks
        self.title = title
        self.show_diagonal = show_diagonal
        self.show_axes = show_axes   
        self.square = square
        self.add_best_fit_lines=add_best_fit_lines
        
    def create(self, data, new_fig=True, \
               figsize=None, minimal_labels=False, **kwargs):
    
        if new_fig:
            if figsize is None:
                figsize=(10,10)
            plt.figure(figsize=figsize)
            
        subject_inds = kwargs['subject_inds'] if 'subject_inds' in kwargs.keys() else None
        
        if subject_inds is not None:
            subject_inds = copy.deepcopy(subject_inds).astype('int')
            assert(np.all(subject_inds>=0))
            n_color_groups = np.max(subject_inds)+1
        else:
            subject_inds = np.zeros((data.shape[0],))
            n_color_groups = 1
            
        if self.color is None:
            if subject_inds is not None:
                color = cm.tab10(np.linspace(0,1,n_color_groups))
            else:
                color = [0.29803922, 0.44705882, 0.69019608, 1]
        else:
            # if i specify n colors and there are fewer groups represented here, 
            # this is ok (will keep colors consistent even if a subj is missing)
            assert(self.color.shape[0]>=n_color_groups)
            color = self.color
            
        for gg in range(n_color_groups):
            inds = (subject_inds==gg);
            plt.plot(data[inds,0], data[inds,1],'.',color=color[gg,:])
                                
        if self.add_best_fit_lines:
            # quick linear regression to get a best fit line
            X = np.concatenate([data[inds,0:1], np.ones((data.shape[0],1))], axis=1)
            y = data[:,1:2]
            linefit =  np.linalg.pinv(X) @ y           
            yhat = data[:,0]*linefit[0] + linefit[1]
            plt.plot(data[:,0], yhat, '-', color=[0.6, 0.6, 0.6])
            
        if self.square==True:
            plt.axis('square')
        
        if not minimal_labels:
            if self.xlabel is not None:
                plt.xlabel(self.xlabel)
            if self.ylabel is not None:
                plt.ylabel(self.ylabel)
            if self.xticks is not None:
                plt.xticks(self.xticks)
            if self.yticks is not None:
                plt.yticks(self.yticks)
        else:
            plt.yticks([])
            plt.xticks([])
            
        if self.title is not None:
            plt.title(self.title)
        if self.xlims is not None:
            plt.xlim(self.xlims)
        if self.ylims is not None:
            plt.ylim(self.ylims)
            
        if self.show_diagonal:
            if self.xlims is None:
                xlim = plt.gca().get_xlim()
            else:
                xlim = self.xlims
            if self.ylims is None:
                ylim = plt.gca().get_ylim()
            else:
                ylim = self.ylims
            
            plt.plot(xlim,ylim, color=[0.8, 0.8, 0.8])
        if self.show_axes:
            plt.axvline(color=[0.8, 0.8, 0.8])
            plt.axhline(color=[0.8, 0.8, 0.8])
            
class violin_plot:
       
    """
    A violin plot object that plots the distribution of each column in a data matrix.
    
    Can be passed to "create_roi_subplots" to draw many violin plots.
    
    colors: optional array of colors, [n_violins x 3]
    column_labels: optional list of strings, [n_violins]
    ylabel: optional, string for yaxis label
    yticks: optional, list of y-axis ticks to use
    ylims: optional, tuple of y-lims
    title: optional title for the violin plot
    horizontal_line_pos: optional, position to draw a horizontal line on the plot    
    
    Use "create" method to pass in data and make the plot.
    data: [n_samples x n_violins], each violin will be one column.
    new_fig: boolean, if True create a new figure, if False assume we already have a figure open.
    figsize: optional figure size
    minimal_labels: boolean, if True the plot will not have xticks/yticks/xlims/ylims
    
    """
        
    def __init__(self, colors=None, column_labels=None, ylabel=None, yticks=None, \
                 title=None, horizontal_line_pos=None, ylims=None):
        
        self.colors = colors
        self.column_labels = column_labels
        self.ylabel = ylabel
        self.title = title
        self.horizontal_line_pos = horizontal_line_pos
        self.ylims = ylims
        self.yticks = yticks
        
    def create(self, data, new_fig=True, figsize=None, minimal_labels=False, **kwargs):
    
        if new_fig:
            if figsize is None:
                figsize=(16,8)
            plt.figure(figsize=figsize)

        n_columns = data.shape[1]
        if self.colors is None:
            colors = cm.plasma(np.linspace(0,1,n_columns))
            colors = np.flipud(colors)
        else:
            colors = self.colors
            
        if data.shape[0]>0:
            for cc in range(n_columns):

                parts = plt.violinplot(data[:,cc], [cc])
                for pc in parts['bodies']:
                    pc.set_color(self.colors[cc,:])
                parts['cbars'].set_color(self.colors[cc,:])
                parts['cmins'].set_color(self.colors[cc,:])
                parts['cmaxes'].set_color(self.colors[cc,:])

        if not minimal_labels:
            if self.column_labels is not None:
                plt.xticks(ticks=np.arange(0,n_columns),labels=self.column_labels,\
                           rotation=45, ha='right',rotation_mode='anchor')
            if self.ylabel is not None:
                plt.ylabel(self.ylabel)
            if self.yticks is not None:
                plt.yticks(self.yticks)
        else:
            plt.xticks([])
            plt.yticks([])
            
        if self.title is not None:
            plt.title(self.title)
        if self.horizontal_line_pos is not None:
            plt.axhline(self.horizontal_line_pos, color=[0.8, 0.8, 0.8])
        if self.ylims is not None:
            plt.ylim(self.ylims)
            

def plot_multi_bars(
    mean_data,
    err_data=None,
    point_data=None,
    add_ss_lines=False,
    colors=None,
    space=0.3,
    space_inner = 0, 
    xticklabels=None,
    ylabel=None,
    ylim=None,
    horizontal_line_pos=0,
    title=None,
    legend_labels=None,
    legend_overlaid=False,
    legend_separate=True,
    add_brackets=None,
    bracket_text=None,
    err_capsize=None,
    fig_size=(12, 6),
):

    """Function to create a bar plot with multiple series of data next to each other.
    Allows adding error bars to each bar and adding significance brackets.

    Args:
        mean_data (array): heights of bars to plot; shape [nlevels1 x nlevels2]
            where nlevels1 is the length of each series (i.e. number of clusters of bars),
            and nlevels2 is the number of series (i.e. number of bars per cluster).

        err_data (array, optional): symmetrical error bar lengths, should be same
            size as mean_data.
        colors (array, optional): list of colors, [nlevels2 x 3] (or [nlevels2 x 4]
            if alpha channel)
        space (float, optional): how big of a space between each bar cluster? max is 0.45.
        xticklabels (1d array or list list, optional): name for each bar "cluster",
            should be [nlevels1] in length.
        ylabel (string, optional): yaxis label
        ylim (2-tuple, optional): yaxis limits
        horizontal_line_pos (float, optional): position to draw a horizontal line on plot.
        title (string, optional): title
        legend_labels (list of strings, optional): labels for each series in the
            plot, [nlevels2] length
        legend_overlaid (boolean, optional): want legend drawn windowed on top of the plot?
        legend_separate (boolean, optional): want legend as a separate axis?
        add_brackets (1d array or list of bools, optional): want to draw brackets over each
            pair of bars? This only applies if nlevels2==2. Must be [nlevels1] in length.
        bracket_text (1d array or list of strings, optional): text to draw over
            each bracket (if drawing brackets.) Must be [nlevels1] in length.
        fig_size (2-tuple, optional): size to draw the entire figure

    """
    assert space < 0.45 and space > 0
    assert len(mean_data.shape) == 2
    nlevels1, nlevels2 = mean_data.shape
    if err_data is not None and len(err_data) == 0:
        err_data = None
    if point_data is not None:
        assert(point_data.shape[1]==nlevels1)
        assert(point_data.shape[2]==nlevels2)
        
    edge_pos = [-0.5 + space, 0.5 - space]
    bar_width = (edge_pos[1] - edge_pos[0] - space_inner*(nlevels2-1)) / nlevels2
    offsets = np.linspace(
        edge_pos[0] + bar_width / 2, edge_pos[1] - bar_width / 2, nlevels2
    )
    if colors is None:
        colors = cm.tab10(np.linspace(0, 1, nlevels2))

    fh = plt.figure(figsize=fig_size)
    ax = plt.subplot(1, 1, 1)
    lh = []
    for ll in range(nlevels2):

        h = plt.bar(
            np.arange(nlevels1) + offsets[ll],
            mean_data[:, ll],
            width=bar_width,
            color=colors[ll, :],
        )
        lh.append(h)
        if err_data is not None:
            assert err_data.shape[0] == nlevels1 and err_data.shape[1] == nlevels2
            plt.errorbar(
                np.arange(nlevels1) + offsets[ll],
                mean_data[:, ll],
                err_data[:, ll],
                ecolor="k",
                zorder=20,
                capsize=err_capsize,
                ls="none",
            )
        if point_data is not None:
            for pp in range(point_data.shape[0]):
                plt.plot(np.arange(nlevels1) + offsets[ll], point_data[pp,:,ll], \
                         '.', color=[0.8, 0.8, 0.8], zorder=15)
                
    if add_ss_lines and point_data is not None:
        for ll in range(nlevels1):           
            for pp in range(point_data.shape[0]):
                plt.plot(ll+offsets, point_data[pp,ll,:],'-', color=[0.8, 0.8, 0.8], zorder=15)
                
    if xticklabels is not None:
        assert len(xticklabels) == nlevels1
        plt.xticks(
            np.arange(nlevels1),
            xticklabels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
    if ylim is not None and ylim != []:
        plt.ylim(ylim)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if horizontal_line_pos is not None:
        plt.axhline(horizontal_line_pos, color=[0.8, 0.8, 0.8])
    if title is not None:
        plt.title(title)

    if legend_overlaid and legend_labels is not None:
        assert len(legend_labels) == nlevels2
        ax.legend(lh, legend_labels)

    if add_brackets is not None:
        assert len(add_brackets) == nlevels1
        assert bracket_text is None or len(bracket_text) == nlevels1
        orig_ylim = ax.get_ylim()
        vert_space = 0.02 * (orig_ylim[1] - orig_ylim[0])
        ymax = orig_ylim[1]

        for xx in np.where(add_brackets)[0]:

            # vertical position of the label is always above the bars,
            # or above the x-axis if bars are negative.
            if err_data is not None:
                max_ht = np.max([np.max(mean_data[xx, :] + err_data[xx, :]), 0])
            else:
                max_ht = np.max([np.max(mean_data[xx, :]), 0])
            brack_bottom = max_ht + vert_space * 2
            brack_top = max_ht + vert_space * 3
            text_lab_ht = max_ht + vert_space * 4
            ymax = np.max([ymax, text_lab_ht + vert_space * 3])

            bracket_x1 = np.mean(offsets[0:int(nlevels2/2)])
            bracket_x2 = np.mean(offsets[int(nlevels2/2):])
            
            plt.plot(
                [xx + bracket_x1, xx + bracket_x1, xx + bracket_x2, xx + bracket_x2],
                [brack_bottom, brack_top, brack_top, brack_bottom],
                "-",
                color="k",
                zorder=20
            )

            if bracket_text is not None:
                ax.annotate(
                    bracket_text[xx],
                    xy=(xx, text_lab_ht),
                    zorder=20,
                    color="k",
                    ha="center",
                    fontsize=12,
                )

        if ylim is None or ylim == []:
            # adjust max y limit so text doesn't get cut off.
            plt.ylim([orig_ylim[0], ymax])

    if legend_separate and legend_labels is not None:
        assert len(legend_labels) == nlevels2
        plt.figure()
        for ll in range(nlevels2):
            plt.plot(0, ll, "-", color=colors[ll, :], linewidth=15)
        plt.legend(legend_labels)

    return fh
       