import warnings
warnings.filterwarnings("ignore")
import sys, os, subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib import gridspec
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from pseudotime import smooth_data
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from DensityPlot import density2d

try:
    import seaborn as sns
    seaborn_exists = True
except ImportError:
    print('Seaborn package or dependencies missing')

class MathTextSciFormatter(mticker.Formatter):
    '''
    This formatter can be fed to set ticklabels in scientific notation without
    the annoying "1e" notation (why would anyone do that?).
    Instead, it formats ticks in proper notation with "10^x".

    fmt: the decimal point to keep
    Usage = ax.yaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))
    '''

    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)

class FormatScalarFormatter(mticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        mticker.ScalarFormatter.__init__(self, useOffset=offset,
                                            useMathText=mathText)

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % mticker._mathdefault(self.format)


class scatter():
    def __init__(self,
                 X,
                 c=None,
                 filename='',
                 figsize=(3, 3),
                 cmap=None,
                 s=15,
                 title='',
                 legend_title=None,
                 facecolor='white',
                 xlabel='',
                 ylabel='',
                 plot_cbar=True,
                 alpha=1,
                 fontsize=13,
                 hspace=0.15,
                 readable_titles=False,
                 wspace=0.05,
                 insertion_spacer=0,
                 clusters=None,
                 rotation=0,
                 ncols=5,
                 sort_by_str=None,
                 sina=False,
                 run=True,
                 cbar_ticks=True,
                 density=False,
                 edgecolor=None,
                 density_cmap='Reds',
                 bw='scott',
                 density_alpha=0.6,
                 inverse_dot_order=False,
                 order_cbar_labels=[],
                 adj=None,
                 show_xticks=False,
                 show_yticks=False,
                 ylim=None,
                 xlim=None,
                 legend_loc=None,
                 vmin=None,
                 vmax=None,
                 stratify=False,
                 grid=False,
                 labelsize=12,
                 markerscale=None,
                 bins=300,
                 axis_border='on',
                 force_discrete_cbar=False,
                 **args):
        self.X = X
        if c is None:
            self.c = np.zeros(np.shape(self.X)[0])
            plot_cbar = False
        else:
            if adj is not None:
                assert np.shape(adj)[0] == np.shape(adj)[1],\
                        'Adjacency matrix should have syemmetric length'
                assert np.shape(adj)[0] == np.shape(self.X)[0],\
                        'Adjacency matrix should have the same length as\
                        number of samples'

                if np.ndim(c) == 1:
                    self.c = adj.dot(c) / np.sum(adj, axis=1).reshape(
                        np.shape(c)[0], 1).T[0]
                else:
                    c = np.array(c)
                    self.c = (adj.dot(c.T).T / np.sum(adj, axis=1))
            else:
                self.c = np.array(c)

        self.clusters = clusters
        if self.clusters is not None:
            assert np.shape(clusters)[0] == np.shape(self.X)[0],\
                    'the length of clusters must equal the length of input\
                    samples'

            figsize = (figsize[0], figsize[1] + figsize[0] / 4)
            self.clusters = np.array(clusters)
        self.plot_cbar = plot_cbar
        self.filename = filename
        self.legend_loc = legend_loc
        self.legend_title = legend_title
        if filename != '':
            self.save_plot = True
            dirname = os.path.dirname(os.path.abspath(self.filename))
            if os.path.isdir(dirname) == False:
                os.mkdir(dirname)
        else:
            self.save_plot = False
        self.column_size = int(ncols)
        self.figsize = figsize
        self.cmap = cmap
        self.title = title
        self.facecolor = facecolor
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.alpha = alpha
        self.fontsize = fontsize
        self.insertion_spacer = insertion_spacer
        self.grid = grid
        self.markerscale = markerscale
        self.labelsize = labelsize
        if self.insertion_spacer > 0:
            self.title = make_titles_readable(self.title,
                                    insertion_spacer=self.insertion_spacer)
        self.s = s
        self.hspace = hspace
        self.rotation = rotation
        self.sina = sina
        self.cbar_ticks = cbar_ticks
        self.density = density
        self.bw = bw
        self.density_alpha = density_alpha
        self.fig_transparent = False
        self.inverse_dot_order = inverse_dot_order
        self.order_cbar_labels = order_cbar_labels
        self.edgecolor = edgecolor
        self.show_xticks = show_xticks
        self.show_yticks = show_yticks
        self.bins = bins
        self.args = args
        self.ylim = ylim
        self.xlim = xlim
        self.vmin = vmin
        self.vmax = vmax
        self.stratify = stratify
        self.axis_border = axis_border
        self.force_discrete_cbar = force_discrete_cbar
        if self.facecolor == 'w' or self.facecolor == 'white':
            self.fig_transparent = True
        if self.density == False and density_cmap == 'None':
            self.density_cmap = self.cmap
        else:
            self.density_cmap = density_cmap
        if sort_by_str is not None and self.clusters is not None:
            assert np.in1d(sort_by_str, np.unique(self.clusters)).sum() ==\
                    len(np.unique(self.clusters)), 'The provided sort_by_str\
                    must contain all unique cluster values'

            self.sort_by_str = np.array(sort_by_str)
        else:
            self.sort_by_str = sort_by_str
        if self.plot_cbar == True:
            self.wspace = (wspace + 0.15)
        else:
            self.wspace = wspace
        self.readable_titles = readable_titles
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 1.25
        if run == True:
            self.run()

    def run(self):
        plot_ndim = self.c.ndim
        if plot_ndim == 1:
            if self.force_discrete_cbar == False:
                self.color_dtype = self.c.dtype
            else:
                self.color_dtype = 'str'
            if self.stratify == False:
                return (self.make_single_plot(X=self.X,
                                              c=self.c,
                                              save_plot=self.save_plot))
            else:
                return (self.make_multiple_plot(save_plot=self.save_plot))
        else:
            if self.force_discrete_cbar == False:
                self.color_dtype = self.c[0].dtype
            else:
                self.color_dtype = 'str'
            return (self.make_multiple_plot(save_plot=self.save_plot))

    def make_single_plot(self, X, c, save_plot=False, legend_loc=None):
        fig, ax = plt.subplots(figsize=self.figsize)
        if self.clusters is not None:
            gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
            ax_bottom = plt.subplot(gs[1])
            ax = plt.subplot(gs[0])
            ## We get the old position and shift the bottom plot a bit up to
            ## remove dead space and give more space for the titles
            curr_pos = ax_bottom.get_position()
            curr_pos_top = ax.get_position()
            average_height = (curr_pos.height + curr_pos_top.height) / 2
            ## We set a new position for the bottom plot so that it has more
            ## space between the current top and next top plot
            new_pos = [
                curr_pos.x0,
                (curr_pos.y0 + (self.hspace - 0.05) / 2 * curr_pos.height +
                 (self.hspace - 0.05) / 2 * curr_pos_top.height),
                curr_pos.width, curr_pos.height
            ]
            ax_bottom.set_position(new_pos)
            self.make_bottom_plot(ax_bottom=ax_bottom)
        if self.ylim:
            ax.set_ylim(self.ylim)
        if self.xlim:
            ax.set_xlim(self.xlim)
        scatter_plot(X=X,
                     ax=ax,
                     fig=fig,
                     c=c,
                     xlabel=self.xlabel,
                     ylabel=self.ylabel,
                     fontsize=self.fontsize-2,
                     title=self.title,
                     legend_title=self.legend_title,
                     number_of_plots=1,
                     facecolor=self.facecolor,
                     cmap=self.cmap,
                     s=self.s,
                     alpha=self.alpha,
                     edgecolor=self.edgecolor,
                     plot_cbar=self.plot_cbar,
                     column_size=self.column_size,
                     cbar_ticks=self.cbar_ticks,
                     density=self.density,
                     density_cmap=self.density_cmap,
                     bw=self.bw,
                     density_alpha=self.density_alpha,
                     inverse_dot_order=self.inverse_dot_order,
                     order_cbar_labels=self.order_cbar_labels,
                     show_xticks=self.show_xticks,
                     show_yticks=self.show_yticks,
                     legend_loc=self.legend_loc,
                     vmin=self.vmin,
                     vmax=self.vmax,
                     grid=self.grid,
                     markerscale=self.markerscale,
                     labelsize=self.labelsize,
                     bins=self.bins,
                     axis_border=self.axis_border,
                     color_dtype=self.color_dtype,
                     **self.args)
        if save_plot != False and self.plot_cbar == True:
            fig.savefig(self.filename,
                        bbox_inches='tight',
                        transparent=self.fig_transparent)
        elif save_plot != False and self.plot_cbar == False:
            fig.savefig(self.filename,
                        bbox_inches='tight',
                        transparent=self.fig_transparent)
        else:
            return (fig, ax)

    def make_multiple_plot(self, save_plot=False):
        if self.stratify == False:
            number_of_plots = np.shape(self.c)[0]
        else:
            assert self.color_dtype not in ['int32', 'int64', 'float'],\
                    'must provide categorical color types'
            number_of_plots = len(np.unique(self.c))
        print('Number of plots: %s' % number_of_plots)
        data_array = np.array(self.X)
        if type(self.title) == str:
            new_title = np.repeat(self.title, number_of_plots)
        else:
            new_title = self.title
        assert np.shape(new_title)[0] == number_of_plots,\
        'the number of provided titles does not match that of the\
                provided colors'
        # assert np.shape(data_array)[0] == number_of_plots,\
        # 'the number of provided 2D array does not match that of the\
                # provided colors'

        if self.readable_titles:
            new_title = make_titles_readable(titles=new_title,
                                             insertion_spacer=\
                                             self.insertion_spacer)
        row_size = int(np.ceil(number_of_plots / self.column_size))
        spacer_columns = 0
        new_figsize = ((self.figsize[0]) * (self.column_size + spacer_columns),
                       self.figsize[1] * row_size)
        if self.clusters is None:
            fig, axs = plt.subplots(ncols=self.column_size,
                                    nrows=row_size,
                                    sharex=True,
                                    sharey=True,
                                    figsize=(new_figsize))
            axs = axs.flatten()
            for ax in axs[number_of_plots:]:
                ax.set_visible(False)
            fig.subplots_adjust(wspace=self.wspace, hspace=self.hspace)
        else:
            widths = np.repeat(4, self.column_size)
            heights = np.tile([4, 1], row_size)
            self.hspace *= 2
            fig = plt.figure(figsize=(new_figsize))
            new_grids = gridspec.GridSpec(ncols=self.column_size,
                                          nrows=row_size * 2,
                                          wspace=self.wspace,
                                          width_ratios=widths,
                                          height_ratios=heights,
                                          hspace=self.hspace)
            axs = [fig.add_subplot(x) for x in new_grids]
        count = 0
        row_count_1 = 0
        row_count_2 = self.column_size
        if self.stratify == True:
            to_color = sorted(set(self.c))
        else:
            to_color = self.c
        for c, title in zip(to_color, new_title):
            if self.stratify == True:
                sub = self.c == c
                X = data_array[sub]
                c = np.repeat(c, sub.sum())
            else:
                X = data_array
            if count < number_of_plots:
                if self.clusters is None:
                    ax = axs[count]
                    if self.ylim:
                        ax.set_ylim(self.ylim)
                    if self.xlim:
                        ax.set_xlim(self.xlim)
                    scatter_plot(X=X,
                                 ax=ax,
                                 fig=fig,
                                 c=c,
                                 xlabel=self.xlabel,
                                 ylabel=self.ylabel,
                                 fontsize=self.fontsize-2,
                                 title=title,
                                 legend_title=self.legend_title,
                                 number_of_plots=number_of_plots,
                                 facecolor=self.facecolor,
                                 cmap=self.cmap,
                                 s=self.s,
                                 alpha=self.alpha,
                                 edgecolor=self.edgecolor,
                                 plot_cbar=self.plot_cbar,
                                 column_size=self.column_size,
                                 cbar_ticks=self.cbar_ticks,
                                 density=self.density,
                                 density_cmap=self.density_cmap,
                                 bw=self.bw,
                                 density_alpha=self.density_alpha,
                                 inverse_dot_order=self.inverse_dot_order,
                                 order_cbar_labels=self.order_cbar_labels,
                                 vmin=self.vmin,
                                 vmax=self.vmax,
                                 grid=self.grid,
                                 markerscale=self.markerscale,
                                 labelsize=self.labelsize,
                                 bins=self.bins,
                                 axis_border=self.axis_border,
                                 show_xticks=self.show_xticks,
                                 show_yticks=self.show_yticks,
                                 color_dtype=self.color_dtype)
                else:
                    ax0 = axs[row_count_1]
                    ax1 = axs[row_count_2]
                    ## We get the old position and shift the bottom plot a bit
                    ## up to remove dead
                    ## space and give more space for the titles
                    curr_pos = ax1.get_position()
                    curr_pos_top = ax0.get_position()
                    average_height = (curr_pos.height +
                                      curr_pos_top.height) / 2
                    ## We set a new position for the bottom plot so that it
                    ## has more space between the current top and next top plot
                    new_pos = [
                        curr_pos.x0,
                        (curr_pos.y0 +
                         (self.hspace - 0.05) / 2 * curr_pos.height +
                         (self.hspace - 0.05) / 2 * curr_pos_top.height),
                        curr_pos.width, curr_pos.height
                    ]
                    ax1.set_position(new_pos)
                    scatter_plot(X=X,
                                 ax=ax0,
                                 fig=fig,
                                 c=c,
                                 xlabel=self.xlabel,
                                 ylabel=self.ylabel,
                                 fontsize=self.fontsize-2,
                                 title=title,
                                 legend_title=self.legend_title,
                                 number_of_plots=number_of_plots,
                                 facecolor=self.facecolor,
                                 cmap=self.cmap,
                                 s=self.s,
                                 alpha=self.alpha,
                                 edgecolor=self.edgecolor,
                                 plot_cbar=self.plot_cbar,
                                 column_size=self.column_size,
                                 cbar_ticks=self.cbar_ticks,
                                 density=self.density,
                                 density_cmap=self.density_cmap,
                                 bw=self.bw,
                                 density_alpha=self.density_alpha,
                                 inverse_dot_order=self.inverse_dot_order,
                                 order_cbar_labels=self.order_cbar_labels,
                                 vmin=self.vmin,
                                 vmax=self.vmax,
                                 grid=self.grid,
                                 markerscale=self.markerscale,
                                 labelsize=self.labelsize,
                                 bins=self.bins,
                                 axis_border=self.axis_border,
                                 show_xticks=self.show_xticks,
                                 show_yticks=self.show_yticks,
                                 color_dtype=self.color_dtype)
                    self.make_bottom_plot(ax_bottom=ax1, c_in=c)
                    row_count_1 += 1
                    row_count_2 += 1
                    if row_count_1 % self.column_size == 0:
                        row_count_1 += self.column_size
                        row_count_2 += self.column_size
            count += 1
        ## Set the extra plots to be invisible
        if self.clusters is not None:
            while row_count_1 % self.column_size != 0:
                ax0 = axs[row_count_1]
                ax1 = axs[row_count_2]
                ax0.set_visible(False)
                ax1.set_visible(False)
                row_count_1 += 1
                row_count_2 += 1
        if save_plot != False and self.plot_cbar == True:
            fig.savefig(self.filename,
                        bbox_inches='tight',
                        transparent=self.fig_transparent)
        elif save_plot != False and self.plot_cbar == False:
            fig.savefig(self.filename,
                        bbox_inches='tight',
                        transparent=self.fig_transparent)
        else:
            return (fig, axs)

    def make_bottom_plot(self, ax_bottom, c_in=None):
        '''
        We make a bottom violin plot based on cluster assignment.
        Calculate the median and quantiles of provided values based on
        provided groupings and make a violin plot.
        '''

        if c_in is None:
            c_in = self.c
        if len(np.unique(self.clusters)) == 1:
            color_by_clusters = [self.c]
            c = None
            labels = None
        else:
            color_by_clusters = []
            labels = []
            c = []
            if self.sort_by_str is not None:
                for u in self.sort_by_str:
                    curr_cluster_val = c_in[self.clusters == u]
                    color_by_clusters.append(curr_cluster_val)
                    c.append([np.mean(curr_cluster_val)])
                    labels.append(u)
            else:
                for u in np.unique(self.clusters):
                    curr_cluster_val = c_in[self.clusters == u]
                    color_by_clusters.append(curr_cluster_val)
                    c.append([np.mean(curr_cluster_val)])
                    labels.append(u)
            c = np.array(c)
            c = c / np.max(c)
        make_violin(X=color_by_clusters,
                    ax=ax_bottom,
                    labels=labels,
                    c=c,
                    s=self.s * 3,
                    fontsize=(self.fontsize - 2),
                    cmap=self.cmap,
                    rotation=self.rotation,
                    grid=False,
                    mean=True,
                    sina=self.sina,
                    return_obj=False,
                    vert=True)


def scatter_plot(X,
                 ax,
                 fig,
                 c=None,
                 count=None,
                 xlabel='',
                 ylabel='',
                 fontsize=12,
                 labelsize=12,
                 title='',
                 number_of_plots=1,
                 facecolor='w',
                 cmap=None,
                 s=50,
                 alpha=1,
                 plot_cbar=True,
                 column_size=1,
                 cbar_ticks=True,
                 density=False,
                 density_cmap='hot',
                 bw='scott',
                 density_alpha=0.6,
                 edgecolor=None,
                 inverse_dot_order=False,
                 order_cbar_labels=[],
                 show_xticks=False,
                 show_yticks=False,
                 textlabel=None,
                 connectivity=None,
                 hide_color=False,
                 legend_loc=None,
                 legend_title=None,
                 rasterized=None,
                 vmin=None,
                 vmax=None,
                 grid=False,
                 markerscale=None,
                 bins=300,
                 mode='scatter',
                 axis_border='on',
                 warning = True,
                 color_dtype=None):
    if count is None:
        count = 0
    if c is None:
        c = np.ones(np.shape(X)[0])
    if not color_dtype:
        color_dtype = c.dtype
    if show_yticks == False:
        ax.set_yticks([])
    if show_xticks == False:
        ax.set_xticks([])
    if hide_color == True:
        plot_cbar = False
    if rasterized is None:
        if np.shape(X)[0] > 100:
            rasterized = True
        else:
            rasterized = False
    if axis_border!='sc':
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
    else:
        ax.set_xlabel(xlabel, fontsize=fontsize, x=0.1, ha='left')
        ax.set_ylabel(ylabel, fontsize=fontsize, y=0.1, ha='left')
    ax.set_title(title, fontsize=fontsize)
    ax.set_facecolor(facecolor)
    ax.tick_params(labelsize=labelsize)
    if grid is False:
        ax.grid(False)
    else:
        ax.grid(color='k', alpha=0.6, ls='--')
    if color_dtype in ['float', 'int32', 'int64']:
        if inverse_dot_order == True:
            sorted_id = np.argsort(-c)
        else:
            sorted_id = np.argsort(c)
        if cmap is None:
            cmap = 'viridis'
        artist = ax.scatter(X[sorted_id, 0],
                            X[sorted_id, 1],
                            c=c[sorted_id],
                            cmap=cmap,
                            s=s,
                            alpha=alpha,
                            edgecolor=edgecolor,
                            rasterized=rasterized,
                            vmin=vmin,
                            vmax=vmax)
        if plot_cbar == True:
            fmt = FormatScalarFormatter("%.1f")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(artist, cax=cax, format=fmt)
            if np.max(cbar.get_ticks()) > 100:
                cbar.formatter.set_powerlimits((0, 0))
            cbar.ax.yaxis.set_offset_position('left')
            cbar.ax.tick_params(labelsize=(fontsize - 1))
            if legend_title is not None:
                cbar.ax.set_title(legend_title, fontsize=fontsize)
            cbar.update_ticks()
            if cbar_ticks == False:
                cbar.ax.set_yticklabels([])
    else:
        labels = c
        unique_labels = np.unique(labels)
        if len(unique_labels) > 200 and warning:
            raise ValueError('You have more than 200 unique classes in the color labels, are you sure these are discrete colors?')
        if inverse_dot_order == True:
            unique_labels = unique_labels[::-1]
        if type(cmap) is not dict:
            tick_dictionary = dict([
                (y, x) for x, y in enumerate(sorted(set(unique_labels)))
            ])
            c = np.array([tick_dictionary[x] for x in np.unique(labels)])
            minima = min(c)
            maxima = max(c)
            norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        if hide_color == False:
            if len(order_cbar_labels) > 0:
                for key in order_cbar_labels:
                    assert str(key) in labels, 'Provided sort label %s\
                    is not in the labels' % (str(key))
                    if type(cmap) is dict:
                        assert key in cmap.keys(), 'Provided sort label %s\
                        is not in the colormap dictionary' % (str(key))
                        ax.scatter(x=X[labels == key, 0],
                                   y=X[labels == key, 1],
                                   label=key,
                                   color=cmap[key],
                                   s=s,
                                   alpha=alpha,
                                   edgecolor=edgecolor,
                                   rasterized=rasterized)
                    else:
                        ax.scatter(x=X[labels == key, 0],
                                   y=X[labels == key, 1],
                                   label=key,
                                   color=mapper.to_rgba(tick_dictionary[key]),
                                   s=s,
                                   alpha=alpha,
                                   edgecolor=edgecolor,
                                   rasterized=rasterized)
            else:
                for key in unique_labels:
                    if type(cmap) is dict:
                        assert key in cmap.keys(), 'Provided sort label %s\
                        is not in the colormap dictionary' % (str(key))
                        ax.scatter(x=X[labels == key, 0],
                                   y=X[labels == key, 1],
                                   label=key,
                                   color=cmap[key],
                                   s=s,
                                   alpha=alpha,
                                   edgecolor=edgecolor,
                                   rasterized=rasterized)
                    else:
                        ax.scatter(x=X[labels == key, 0],
                                   y=X[labels == key, 1],
                                   label=key,
                                   color=mapper.to_rgba(tick_dictionary[key]),
                                   s=s,
                                   alpha=alpha,
                                   edgecolor=edgecolor,
                                   rasterized=rasterized)
        if not markerscale:
            markerscale = int(np.sqrt(20 / s) * 3)
        if connectivity is not None:
            lw = s / 10
            central_pos = make_mean_pos(X, clusters=labels)
            for ind, u in enumerate(central_pos.keys()):
                ax.scatter(central_pos[u][0], central_pos[u][1],\
                          s=s*5, cmap='Spectral', c='k')
            draw_edge(central_pos, connectivity=connectivity, ax=ax, max_lw=5)
        if plot_cbar != False and count == (column_size - 1) or\
           plot_cbar != False and count < 3 and count == (number_of_plots-1):
            if legend_loc is not None:
                allowed_loc = [
                    'upper left', 'upper right', 'bottom left', 'bottom right'
                ]
                assert legend_loc in allowed_loc, 'legend_loc must be one of\
                the available choices: {:}'.format(allowed_loc)
                if legend_loc == 'upper left':
                    location = (0.05, 0.95)
                elif legend_loc == 'upper right':
                    location = (0.95, 0.95)
                elif legend_loc == 'bottom left':
                    location = (0.05, 0.05)
                elif legend_loc == 'bottom right':
                    location = (0.95, 0.05)
                lgnd = ax.legend(fontsize=(fontsize - 2),
                                 loc=2,
                                 bbox_to_anchor=location,
                                 fancybox=False,
                                 shadow=False,
                                 edgecolor='black',
                                 frameon=False,
                                 markerscale=markerscale,
                                 borderaxespad=0.)
            else:
                lgnd = ax.legend(fontsize=(fontsize - 2),
                                 loc=2,
                                 bbox_to_anchor=(1.05, 1),
                                 fancybox=False,
                                 shadow=False,
                                 edgecolor='black',
                                 frameon=False,
                                 markerscale=markerscale,
                                 borderaxespad=0.)
            if legend_title is not None:
                lgnd.set_title(legend_title, prop={"size":(fontsize-2)})
    if axis_border != 'on':
        if axis_border == 'off':
            ax.axis('off')
        elif axis_border == 'sc':
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            xdist = (xmax - xmin)/5
            ydist = (ymax - ymin)/5
            ax.spines['bottom'].set_bounds(xmin, xmin+xdist)
            ax.spines['left'].set_bounds(ymin, ymin+ydist)

    if density == True:
        density2d(X, ax, bins=bins, xlim=ax.get_xlim(), ylim=ax.get_ylim(),
                  cmap=density_cmap, mode=mode, mesh_order='top')

def draw_edge(pos_dict, connectivity, ax, max_lw=5):
    keys = list(pos_dict.keys())
    for row, u in enumerate(connectivity):
        for col, width in enumerate(u):
            x1, x2 = pos_dict[keys[row]][0], pos_dict[keys[col]][0]
            y1, y2 = pos_dict[keys[row]][1], pos_dict[keys[col]][1]
            ax.plot([x1, x2], [y1, y2], color='k', lw=width * max_lw)


def make_titles_readable(titles, insertion_spacer=3):
    '''
    This is to shorten gene names and replace characters such as % and _
    It also inserts a linebreaker symbol at every insertion_spacer increment
    '''
    title_v = []
    for i in titles:
        title = str(i).replace('_', ' ').split(' ')
        for eid, i in enumerate(
                range(insertion_spacer - 1,
                      len(title) - 1, insertion_spacer)):
            title[i] += '\n'
        title = ' '.join(title)
        title_v.append(title.rstrip())
    return (np.array(title_v))


def make_mean_pos(X, clusters):
    ## Make central positions based on cluster assignment
    unique_labels = np.unique(clusters)
    central_pos = {}
    for u in unique_labels:
        central_pos[u] = np.array(X[clusters == u].mean(axis=0))
    return central_pos


class heatmap():
    def __init__(self,
                 corr_mat,
                 col_colors=None,
                 row_colors=None,
                 filename='',
                 figsize=(6, 6),
                 method='ward',
                 metric='euclidean',
                 col_cluster=True,
                 row_cluster=True,
                 cmap=None,
                 dpi=300,
                 col_name='Cells',
                 row_name='Genes',
                 col_cmap='jet',
                 row_cmap='jet',
                 xlabel='',
                 ylabel='',
                 title='',
                 fontsize=12,
                 labelsize=10,
                 showyticks=False,
                 showxticks=False,
                 run=True,
                 vmin=None,
                 vmax=None,
                 xticks_top=False,
                 yticks_left=False,
                 show_cbar=True,
                 colorbar_title='',
                 colorbar_position=None):
        self.figsize = figsize
        self.method = method
        self.metric = metric
        self.output = filename
        self.col_cluster = col_cluster
        self.row_cluster = row_cluster
        if cmap is None:
            self.cmap = 'seismic'
        else:
            self.cmap = cmap
        self.dpi = dpi
        self.col_name = col_name
        self.row_name = row_name
        self.col_cmap = col_cmap
        self.row_cmap = row_cmap
        self.title = title
        self.fontsize = fontsize
        self.labelsize = labelsize
        self.font_scale = self.labelsize / 12
        self.showyticks = showyticks
        self.showxticks = showxticks
        self.colorbar_position = colorbar_position
        self.colorbar_title = colorbar_title
        self.vmin = vmin
        self.vmax = vmax
        self.xticks_top = xticks_top
        self.yticks_left = yticks_left
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.show_cbar=show_cbar
        if isinstance(corr_mat, pd.DataFrame) == False:
            self.corr_mat = pd.DataFrame(corr_mat)
        else:
            self.corr_mat = corr_mat
        self.shape = self.corr_mat.shape
        self.cell_names = corr_mat.index.values
        self.gene_names = corr_mat.columns.values
        self.run = run
        if col_colors is None and row_colors is None:
            self.draw_without_colors()
        else:
            if col_colors is None:
                self.col_colors = []
            else:
                self.col_colors = col_colors
            if row_colors is None:
                self.row_colors = []
            else:
                self.row_colors = row_colors
            self.draw_with_colors()

    def draw_without_colors(self):
        sns.set(font_scale=self.font_scale)
        self.g = sns.clustermap(self.corr_mat.T,
                                method=self.method,
                                metric=self.metric,
                                figsize=self.figsize,
                                row_cluster=self.row_cluster,
                                col_cluster=self.col_cluster,
                                vmin=self.vmin,
                                vmax=self.vmax,
                                cmap=self.cmap,
                                **{'xticklabels':True, 'yticklabels':True})
        ax = self.g.ax_heatmap
        if self.show_cbar == False:
            self.g.cax.set_visible(False)
        else:
            self.g.cax.set_position((0.15, 0.17, 0.03, 0.45))
            self.g.cax.yaxis.tick_left()
        if self.showxticks == False:
            ax.set_xticklabels([])
        if self.showyticks == False:
            ax.set_yticks([])
        ax.tick_params(axis='y', rotation=0)
        ax.set_title(self.title, fontsize=self.fontsize)
        self.fixedWidthClusterMap()
        if self.output != '' and self.run == True:
            self.g.savefig(self.output, dpi=self.dpi)
        if self.run == False:
            return self.g

    def draw_with_colors(self):
        sns.set(font_scale=self.font_scale)
        col_linkage = hierarchy.linkage(self.corr_mat.T,
                                        method=self.method,
                                        metric=self.metric)
        row_linkage = hierarchy.linkage(self.corr_mat,
                                        method=self.method,
                                        metric=self.metric)
        if np.shape(self.col_colors)[0] > 0:
            if np.ndim(self.col_colors) == 1:
                self.col_colors_series = self.return_color_series(
                    colors=self.col_colors,
                    index=self.cell_names,
                    col_name=self.col_name,
                    cmap=self.col_cmap)
            else:
                self.col_colors_series = pd.DataFrame(index=self.cell_names)
                if np.ndim(self.col_cmap) == 0:
                    self.col_cmap = np.repeat(self.col_cmap,
                                              np.ndim(self.col_colors))
                elif np.shape(self.col_cmap)[0] != np.ndim(self.col_colors):
                    self.col_cmap = np.repeat(self.col_cmap[0],
                                              np.ndim(self.col_colors))
                assert np.ndim(self.col_name) > 0,\
                        'Provided column color names must be a list or array'
                assert np.shape(self.col_name)[0] ==\
                        np.shape(self.col_colors)[0],\
                        'The number of provided column colors labels does not\
                match length of provided color list'

                for color, col_name, col_cmap in zip(self.col_colors,
                                                     self.col_name,
                                                     self.col_cmap):
                    self.col_colors_series[col_name]=\
                            self.return_color_series(colors=color,
                                                     index=self.cell_names,
                                                     col_name=col_name,
                                                     cmap=col_cmap).values
        if np.shape(self.row_colors)[0] > 0:
            if np.ndim(self.row_colors) == 1:
                self.row_colors_series=\
                        self.return_color_series(colors=self.row_colors,
                                                 index=self.gene_names,
                                                 col_name=self.row_name,
                                                 cmap=self.row_cmap)
            else:
                self.row_colors_series = pd.DataFrame(index=self.gene_names)
                if np.ndim(self.row_cmap) == 0:
                    self.row_cmap = np.repeat(self.row_cmap,
                                              np.ndim(self.row_colors))
                elif np.shape(self.row_cmap)[0] != np.ndim(self.row_colors):
                    self.row_cmap = np.repeat(self.row_cmap[0],
                                              np.ndim(self.row_colors))
                assert np.ndim(self.row_name) > 0,\
                        'Provided row color names must be a list or array'
                assert np.shape(self.row_name)[0] == np.ndim(self.row_colors),\
                        'The number of provided row colors labels does not\
                        match length of provided color list'

                for color, row_name, row_cmap in zip(self.row_colors,
                                                     self.row_name,
                                                     self.row_cmap):
                    self.row_colors_series[row_name]=\
                            self.return_color_series(colors=color,
                                                     index=self.gene_names,
                                                     col_name=row_name,
                                                     cmap=row_cmap).values
        if np.shape(self.col_colors)[0] > 0 and\
           np.shape(self.row_colors)[0] > 0:
            self.g = sns.clustermap(self.corr_mat.T,
                                    method=self.method,
                                    metric=self.metric,
                                    figsize=self.figsize,
                                    row_cluster=self.row_cluster,
                                    col_cluster=self.col_cluster,
                                    col_colors=self.col_colors_series,
                                    row_colors=self.row_colors_series,
                                    vmin=self.vmin,
                                    vmax=self.vmax,
                                    cmap=self.cmap,
                                    linewidths=0,
                                    **{'xticklabels':True, 'yticklabels':True})
            if type(self.row_name) == str:
                self.g.ax_row_colors.set_xticklabels([self.row_name],
                                                     rotation=0,
                                                     ha='right')
            else:
                if np.shape(self.row_name)[0] > 1:
                    rot = 45
                else:
                    rot = 0
                self.g.ax_row_colors.set_xticklabels(self.row_name,
                                                     rotation=rot,
                                                     ha='right')
        elif np.shape(self.col_colors)[0] > 0:
            self.g = sns.clustermap(self.corr_mat.T,
                                    method=self.method,
                                    metric=self.metric,
                                    figsize=self.figsize,
                                    row_cluster=self.row_cluster,
                                    col_cluster=self.col_cluster,
                                    col_colors=self.col_colors_series,
                                    vmin=self.vmin,
                                    vmax=self.vmax,
                                    cmap=self.cmap,
                                    **{'xticklabels':True, 'yticklabels':True})
        elif np.shape(self.row_colors)[0] > 0:
            self.g = sns.clustermap(self.corr_mat.T,
                                    method=self.method,
                                    metric=self.metric,
                                    figsize=self.figsize,
                                    row_cluster=self.row_cluster,
                                    col_cluster=self.col_cluster,
                                    row_colors=self.row_colors_series,
                                    vmin=self.vmin,
                                    vmax=self.vmax,
                                    cmap=self.cmap,
                                    **{'xticklabels':True, 'yticklabels':True})
            if type(self.row_name) == str:
                self.g.ax_row_colors.set_xticklabels([self.row_name],
                                                     rotation=0,
                                                     ha='right')
            else:
                if np.shape(self.row_name)[0] > 1:
                    rot = 45
                else:
                    rot = 0
                self.g.ax_row_colors.set_xticklabels(self.row_name,
                                                     rotation=rot,
                                                     ha='right')
        if self.show_cbar == False:
            self.g.cax.set_visible(False)
        else:
            if self.colorbar_position:
                self.g.cax.set_position(self.colorbar_position)
            else:
                if self.showyticks == False:
                    self.g.cax.set_position((0.15, 0.17, 0.03, 0.45))
                self.g.cax.set_position((0.15, 0.17, 0.03, 0.45))
            self.g.cax.yaxis.tick_left()
            self.g.cax.set_title(self.colorbar_title,
                                 fontsize=self.fontsize-2)
        plt.draw()
        ax = self.g.ax_heatmap
        ax.set_title(self.title, fontsize=self.fontsize, y=1.1)
        ax.tick_params(axis='y', rotation=0)
        self.fixedWidthClusterMap()
        if self.xticks_top == True:
            ax.xaxis.tick_top()
        if self.yticks_left == True:
            ax.yaxis.tick_left()
        if self.showxticks == False:
            ax.set_xticklabels([])
        if self.showyticks == False:
            ax.set_yticks([])
        if self.output != '' and self.run == True:
            self.g.savefig(self.output, dpi=self.dpi, bbox_inches='tight')
        if self.run == False:
            return self.g

    def return_color_series(self, colors, index, col_name, cmap='jet'):
        if type(cmap) == type(sns.color_palette()):
            cmap = ListedColormap(cmap.as_hex(), name='custom_cmap')
        colors = np.array(colors)
        unique_val = np.unique(colors)
        unique_val_num = np.linspace(0, 1, len(unique_val))
        if type(cmap) != dict:
            unique_colors = [
                plt.get_cmap(cmap)(c)
                for x, c in zip(unique_val, unique_val_num)
            ]
        else:
            unique_colors = [cmap[x] for x in unique_val]
        c_lut = dict(zip(unique_val, unique_colors))
        color_series = pd.Series(colors).map(c_lut)
        color_series.index = index
        color_series.rename(col_name, inplace=True)
        return color_series

    def make_labels(self):
        self.g.ax_heatmap.set_xlabel(self.xlabel)
        self.g.ax_heatmap.set_ylabel(self.ylabel)

    def fixedWidthClusterMap(self, cellSizePixels=50):
        # calculate the size of the heatmap axes
        dpi = mpl.rcParams['figure.dpi']
        marginWidth = mpl.rcParams['figure.subplot.right']-mpl.rcParams['figure.subplot.left']
        marginHeight = mpl.rcParams['figure.subplot.top']-mpl.rcParams['figure.subplot.bottom']
        Nx, Ny = self.shape
        figWidth = (Nx*cellSizePixels/dpi)/1/marginWidth
        figHeight = (Ny*cellSizePixels/dpi)/1/marginHeight

        axWidth = (Nx*cellSizePixels)/(figWidth*dpi)
        axHeight = (Ny*cellSizePixels)/(figHeight*dpi)
        shrink_dendrogram_x = (np.log2(figWidth) / np.log2(figHeight))*1.1
        shrink_dendrogram_y = (1 / shrink_dendrogram_x)*1.1

        # resize heatmap
        ax_heatmap_orig_pos = self.g.ax_heatmap.get_position()
        # self.g.ax_heatmap.set_position([ax_heatmap_orig_pos.x0, ax_heatmap_orig_pos.y0,
                                      # axWidth, axHeight])

        # resize dendrograms to match
        ax_row_orig_pos = self.g.ax_row_dendrogram.get_position()
        # self.g.ax_row_dendrogram.set_position([ax_row_orig_pos.x0*shrink_dendrogram_x, ax_row_orig_pos.y0,
                                             # ax_row_orig_pos.width/shrink_dendrogram_x, axHeight])
        ax_col_orig_pos = self.g.ax_col_dendrogram.get_position()
        # self.g.ax_col_dendrogram.set_position([ax_col_orig_pos.x0, ax_heatmap_orig_pos.y0+axHeight,
                                             # axWidth, ax_col_orig_pos.height/shrink_dendrogram_y])

    def run(self):
        self.zero_out_diag()
        self.make_graph()
        self.generate_pos()
        self.make_labels()
        return self.g


def set_axis_style(ax, labels, fontsize=12, rotation=0):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(0, len(labels)))
    ax.set_xticklabels(labels, fontsize=fontsize, rotation=rotation)


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return (lower_adjacent_value, upper_adjacent_value)


def gene_expression_plot(X, figsize=(4, 3), genes=None, ncols=1,
                         sep_by_group=[], sharex=False, sharey=False,
                         fontsize=12, legend_loc=(1.0, 0.7), **args):
    size = np.shape(X)[0]
    if isinstance(X, pd.DataFrame):
        if genes is None:
            genes = X.index.values
        else:
            genes = genes
        D = X.values.copy()
    else:
        if genes is None:
            genes = np.arange(size)
        D = X.copy()
    nrows = int(np.ceil(size / ncols))
    print('expected number of plots: {:}'.format(size))
    if len(sep_by_group) > 0:
        sep_by_group = np.array(sep_by_group)
        ugroups = np.unique(sep_by_group)
        sep_plots = True
    else:
        sep_plots = False
    assert size < 100, 'Cannot make more than 100 plots'
    fig, axs = plt.subplots(figsize=(figsize[0]*ncols,
                                     figsize[1]*nrows), ncols=ncols,
                            nrows=nrows, sharex=sharex, sharey=sharey)
    if sharex is False:
        fig.subplots_adjust(hspace=0.25)
    axs = axs.flatten()
    for i, ax in enumerate(axs[:size]):
        if sep_plots:
            for j, u in enumerate(ugroups):
                span = D[i].max() - D[i].min()
                to_plot = D[i][sep_by_group == u]
                if len(to_plot) > 1:
                    sns.kdeplot(to_plot, ax=ax, label=u, color='C'+str(j),
                                **args)
                else:
                    ax.bar(to_plot, 1, width=np.sqrt(span)/4, color='C'+str(j),
                           label=u, alpha=0.7, edgecolor='None')
                if i == (ncols-1):
                    ax.legend(fontsize=fontsize, loc=legend_loc)
                else:
                    ax.legend().set_visible(False)
        else:
            sns.kdeplot(D[i], ax=ax, **args)
        ax.set_title(genes[i])
        ax.set_ylabel('Counts', x=-1)
    for ax in axs[size:]: ax.set_visible(False)
    return fig, axs

class violin():
    def __init__(self,
                 X,
                 labels=None,
                 c=None,
                 figsize=(4, 3),
                 filename='',
                 cmap='Set2',
                 s=50,
                 title='',
                 fontsize=12,
                 alpha=1,
                 rotation=0,
                 ylabel='',
                 xlabel='',
                 grid=False,
                 mean=False,
                 show_yticks=True,
                 sina=False,
                 random_seed=False,
                 facecolor='white',
                 run=True,
                 vert=True,
                 logy=False):
        if random_seed == False:
            np.random.seed(0)
        self.dataframe = False
        if isinstance(X, pd.DataFrame):
            if labels is None:
                labels = X.columns.values
            self.X = X.values.T
            self.dataframe = True
        elif isinstance(X, list):
            self.X = X
        else:
            self.X = X.T
        self.figsize = figsize
        self.filename = filename
        self.cmap = cmap
        self.s = s
        self.title = title
        self.fontsize = fontsize
        self.alpha = alpha
        self.rotation = rotation
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.grid = grid
        self.mean = mean
        self.show_yticks = show_yticks
        self.sina = sina
        self.vert = vert
        self.facecolor = facecolor
        self.logy = logy
        if filename != '':
            self.save_plot = True
            dirname = os.path.dirname(os.path.abspath(self.filename))
            if os.path.isdir(dirname) == False:
                os.mkdir(dirname)
        else:
            self.save_plot = False
        if self.facecolor == 'w' or self.facecolor == 'white':
            self.fig_transparent = True
        else:
            self.fig_transparent = False
        if np.ndim(self.X) == 1:
            self.ndim = 1
        else:
            self.ndim = np.shape(self.X)[0]
        if labels is None:
            self.labels = np.arange(0, np.shape(self.X)[0])
        else:
            assert isinstance(labels, (list, np.ndarray)), 'Provided labels\
                    must be either numpy array or list'

            if np.ndim(self.X) == 1:
                assert np.shape(labels)[0] == np.shape(self.X)[0], 'You\
                        provided more labels than there are for number of rows\
                        in X'

                self.labels = labels
            else:
                assert np.shape(self.X)[0] == np.shape(labels)[0], 'Number of\
                        labels does not match the length of rows in X'

                self.labels = labels
        if c is None:
            self.c = np.ones(np.shape(self.X)[0])
        else:
            assert isinstance(c, (list, np.ndarray)), 'Provided labels must be\
                    either numpy array or list'

            if np.ndim(self.X) == 1:
                assert np.shape(c)[0] == np.shape(X)[0], 'You provided more\
                        labels than there are for number of rows in X'

                self.c = c
            else:
                assert np.shape(self.X)[0] == np.shape(c)[0], 'Number of\
                        labels does not match the length of rows in X'

                self.c = c
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 1.25
        if run == True:
            self.run()

    def run(self):
        return (make_violin(self.X,
                            figsize=self.figsize,
                            c=self.c,
                            fontsize=self.fontsize,
                            rotation=self.rotation,
                            s=self.s,
                            cmap=self.cmap,
                            xlabel=self.xlabel,
                            ylabel=self.ylabel,
                            facecolor=self.facecolor,
                            labels=self.labels,
                            grid=self.grid,
                            mean=self.mean,
                            show_yticks=self.show_yticks,
                            title=self.title,
                            sina=self.sina,
                            filename=self.filename,
                            transparent=self.fig_transparent,
                            logy=self.logy,
                            vert=self.vert))


def make_violin(X,
                figsize=(4, 4),
                ax=None,
                labels=None,
                c=None,
                fontsize=12,
                rotation=0,
                s=50,
                cmap='gnuplot',
                xlabel='',
                ylabel='',
                grid=False,
                mean=False,
                show_yticks=False,
                title='',
                sina=False,
                sample_threshold=50,
                random_seed=True,
                facecolor='white',
                filename='',
                transparent=True,
                return_obj=True,
                vert=True,
                logy=False):
    X = np.array(X)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    number_of_samples = np.shape(X)[0]
    if c is None:
        if number_of_samples == 1:
            c = [1]
        else:
            c = np.ones(np.shape(X)[0])
    if number_of_samples == 1:
        inds = [0]
    else:
        inds = np.arange(0, np.shape(X)[0])
    if labels is None:
        labels = inds
    if title != '':
        ax.set_title(title, fontsize=fontsize, y=1.02)
    ax.tick_params(labelsize=fontsize)
    quartile1 = np.zeros(number_of_samples)
    medians = quartile1.copy()
    quartile3 = quartile1.copy()
    quant_median = [np.percentile(x, [25, 50, 75]) for x in X]
    mean_v = np.array([np.nanmean(x) for x in X])
    for eid, i in enumerate(quant_median):
        quartile1[eid] = i[0]
        medians[eid] = i[1]
        quartile3[eid] = i[2]
    whiskers = np.array([adjacent_values(sorted_array, q1, q3) for \
                         sorted_array, q1, q3 in zip(X, quartile1, quartile3)])
    whiskersMin, whiskersMax = (whiskers[:, 0], whiskers[:, 1])
    ax.set_facecolor(facecolor)
    ax.grid(False)
    if sina == False:
        if mean == True:
            ax.scatter(inds,
                       mean_v,
                       marker='o',
                       color='white',
                       edgecolor='k',
                       s=s,
                       zorder=3,
                       lw=1)
        else:
            ax.scatter(inds,
                       medians,
                       marker='o',
                       color='white',
                       edgecolor='k',
                       s=s,
                       zorder=3,
                       lw=1)
    else:
        x_inds = []
        swarmplot_y = []
        for i, j in zip(X, inds):
            nm = np.shape(i)[0]
            if nm > sample_threshold:
                swarmplot_y.append(
                    np.random.choice(i, sample_threshold, replace=False))
                x_inds.append(np.repeat(j, sample_threshold))
            else:
                swarmplot_y.append(i)
                x_inds.append(np.repeat(j, nm))
        swarmplot_y = np.concatenate(swarmplot_y)
        sns.swarmplot(x=np.concatenate(x_inds),
                      y=swarmplot_y,
                      ax=ax,
                      color='k',
                      s=int(s / 10),
                      alpha=0.7)
    parts = ax.violinplot([x for x in X],
                          positions=inds,
                          showmeans=False,
                          showmedians=False,
                          showextrema=False,
                          vert=vert)
    for pc, color in zip(parts['bodies'], c):
        pc.set_facecolor(plt.get_cmap(cmap)(color))
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
        pc.set_alpha(1)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=3)
    ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
    set_axis_style(ax, labels, fontsize=fontsize, rotation=rotation)
    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=5)
    if logy == True:
        ax.set_yscale('log')
        ymin = (np.max([np.max(x) for x in X]))
        ymax = (np.min([np.min(x) for x in X]))
        if ymax < 1:
            yticks = [x for x in ax.get_yticklabels()]
        # ax.set_yticklabels(yticks)
    if show_yticks == False:
        ax.set_yticks([])
    if grid == True:
        ax.yaxis.grid(color='k', linestyle=':', lw=1)
    if filename != '':
        fig.savefig(filename, bbox_inches='tight', transparent=transparent)
    elif return_obj == True:
        return (fig, ax)


def fancy_dendrogram(Z,
                     ax=None,
                     xlabel='',
                     ylabel='',
                     figsize=(4, 4),
                     **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    ddata = dendrogram(Z, **kwargs)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if not kwargs.get('no_plot', False):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'],
                           ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                ax.plot(x, y, 'o', c=c)
                ax.annotate("%.3g" % y, (x, y),
                            xytext=(0, -5),
                            textcoords='offset points',
                            va='top',
                            ha='center')
        if max_d:
            ax.axhline(y=max_d, c='k')
    return ddata


def pairplot(data, figsize=(4, 4), c=[], cmap='viridis', **kwargs):
    if isinstance(data, pd.DataFrame) is False:
        data = pd.DataFrame(data)
    cmap = 'viridis'
    g = sns.pairplot(data, **kwargs)
    mask = (np.ones(np.shape(g.axes), dtype=bool))
    np.fill_diagonal(mask, False)
    axs = g.axes[mask].flatten()
    if len(c) > 0:
        for ax in axs:
            ax.grid(False)
            data = ax.collections[0].get_offsets()
            del (ax.collections[0])
            ax.scatter(data[:, 0], data[:, 1], c=c, s=10, cmap=cmap)
    plt.close()
    return g.fig


class line_scatter():
    def __init__(self,
                 X,
                 ncols=4,
                 filename='',
                 fontsize=12,
                 s=10,
                 labelpad=10,
                 insertion_spacer=0,
                 hspace=0.5,
                 wspace=0,
                 lw=2,
                 show_xticks=False,
                 show_yticks=False,
                 xlabel='',
                 figsize=(3, 1),
                 titles=[],
                 nonzero=False,
                 smooth_k=3,
                 c=[],
                 cmap='tab20b',
                 line_color='b'):
        if isinstance(X, pd.DataFrame) and titles == []:
            titles = X.columns.values
        self.X = np.array(X)
        self.smooth_X, residuals = smooth_data(self.X, k=smooth_k)
        self.nonzero = nonzero
        if self.nonzero == False:
            self.smooth_X[self.smooth_X < 0] = 0
        self.ncols = ncols
        self.filename = filename
        self.fontsize = fontsize
        self.s = s
        self.labelpad = labelpad
        self.insertion_spacer = insertion_spacer
        self.hspace = hspace
        self.wspace = wspace
        self.show_xticks = show_xticks
        self.figsize = figsize
        self.titles = titles
        self.show_xticks = show_xticks
        self.show_yticks = show_yticks
        self.xlabel = xlabel
        self.lw = lw
        self.cmap = cmap
        self.line_color = line_color
        if np.shape(c)[0] == 0:
            self.c = 'k'
        else:
            self.c = np.array(c)
        if np.ndim(self.X) == 1:
            self.num_of_plots = 1
        else:
            self.num_of_plots = np.shape(self.X)[1]
        self.run()

    def run(self):
        fig, axs = plt.subplots(
            figsize=(self.figsize[0] * self.ncols,
                     self.figsize[1] * self.num_of_plots / self.ncols),
            nrows=int(np.ceil(self.num_of_plots / self.ncols)),
            ncols=self.ncols,
            sharex=True)
        axs = axs.flatten()
        fig.subplots_adjust(hspace=self.hspace, wspace=self.wspace)
        x_val = np.arange(self.X.shape[0])
        unique_labels, index = np.unique(self.c, return_inverse=True)
        for ax, col_id in zip(axs[:self.num_of_plots],
                              np.arange(self.num_of_plots)):
            ax.scatter(x_val,
                       self.X[:, col_id],
                       s=self.s,
                       c=index,
                       cmap=self.cmap)
            ax.plot(x_val,
                    self.smooth_X[:, col_id],
                    lw=self.lw,
                    c=self.line_color)
            if self.titles != []:
                ax.set_title(self.titles[col_id],
                             fontsize=self.fontsize,
                             y=1.02)
            if self.show_yticks == False:
                ax.set_yticklabels([])
            if self.show_xticks == False:
                ax.set_xticklabels([])
        for ax in axs[self.num_of_plots:]:
            ax.set_visible(False)
        if self.filename != '':
            fig.savefig(self.filename, bbox_inches='tight')
        else:
            return (fig, axs)


def make_gif(filenames, output, fname='myimage', remove_png=True):
    full_input_names = ' '.join([output + x for x in filenames])
    command = '/usr/bin/convert -delay 10 -loop 0 ' + full_input_names + ' ' +\
            output + fname  + '.gif'
    subprocess.call(command, shell=True)
    if remove_png == True:
        command = 'rm ' + full_input_names
        subprocess.call(command, shell=True)


def animate(ax,
            filename,
            angle=0,
            fname='rotate',
            dpi=200,
            interval=5,
            start_x=30,
            start_y=None,
            style=1,
            clockwise=True):
    filenames = []
    start_count = 0
    angle = start_x
    if clockwise is True:
        direction = -1
    else:
        direction = 1
    if style == 1 or style == 2:
        if start_y is None:
            start_y = 0
        for eid, angle in enumerate(range(start_x, start_x + 360,
                                          interval)[::direction]):
            ax.view_init(start_y, angle)
            plt.draw()
            f = fname + '_' + str(start_count) + '.png'
            plt.savefig(filename + f, bbox_inches='tight', dpi=dpi)
            filenames.append(f)
            start_count += 1
        if start_y is None:
            start_y = start_x
    if start_y is None:
        start_y = 0
    if style == 2 or style == 3:
        for eid, yangle in enumerate(range(start_y, start_y + 110,
                                           interval)[::direction]):
            ax.view_init(yangle, angle)
            plt.draw()
            f = fname + '_' + str(start_count) + '.png'
            plt.savefig(filename + f, bbox_inches='tight', dpi=dpi)
            filenames.append(f)
            start_count += 1
        for eid, yangle in enumerate( range(start_y, start_y + 110,
                                            interval)[::-direction]):
            ax.view_init(yangle, angle)
            plt.draw()
            f = fname + '_' + str(start_count) + '.png'
            plt.savefig(filename + f, bbox_inches='tight', dpi=dpi)
            filenames.append(f)
            start_count += 1
    make_gif(filenames, filename, fname)


class dotplot():
    def __init__(self,
                 df_size,
                 df_color=[],
                 df_shape=[],
                 figsize=(3, 4),
                 max_size=150,
                 x_spacing=None,
                 y_spacing=None,
                 x_rotation=0,
                 y_rotation=0,
                 fontsize=12,
                 ticksize=10,
                 edgecolor='k',
                 lw=1,
                 c='',
                 shape='o',
                 filename='',
                 run=True,
                 verbose=False,
                 cbar_ticks=True,
                 legend_title='',
                 frameon=True,
                 legend_ticks=4,
                 cmap='coolwarm',
                 colorbar_title='',
                 xlabel='',
                 ylabel='',
                 round_point=2):
        self.figsize = figsize
        self.max_size = max_size
        self.filename = filename
        self.nrows = np.shape(df_size)[1]
        self.ncols = np.shape(df_size)[0]
        self.x_rotation = x_rotation
        self.y_rotation = y_rotation
        self.c = c
        self.cmap = cmap
        self.shape = shape
        self.dot_color = np.array(df_color).T
        self.dot_shape = np.array(df_shape).T
        self.verbose = verbose
        self.legend_ticks = legend_ticks
        self.legend_title = legend_title
        self.frameon = frameon
        self.colorbar_title = colorbar_title
        self.fontsize = fontsize
        self.ticksize = ticksize
        self.cbar_ticks = cbar_ticks
        self.round_point = round_point
        self.xlabel = xlabel
        self.ylabel = ylabel
        assert self.dot_color.dtype in [
            'float64', 'float32', 'float', 'int64', 'int32', 'int'
        ], 'Provided dot color\
                                        values must be numeric'

        self.edgecolor = edgecolor
        self.lw = lw
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing
        if isinstance(df_size, pd.DataFrame):
            self.xticklabels = df_size.index.values
            self.yticklabels = df_size.columns.values
            self.dot_size = np.array(df_size)
        elif isinstance(df_size, pd.core.series.Series):
            self.xticklabels = df_size.index.values
            self.yticklabels = np.arange(np.shape(df_size)[1])
            self.dot_size = np.array(df_size)
        else:
            self.dot_size = np.array(df_size)
            self.xticklabels = np.arange(np.shape(df_size)[0])
            self.yticklabels = np.arange(np.shape(df_size)[1])
        self.dot_size_values = self.dot_size.copy()
        self.dot_size = self._normalize_dot_size(self.dot_size)
        if self.x_spacing is None or self.y_spacing is None:
            if self.x_spacing is None and self.y_spacing is not None:
                if np.shape(self.dot_size)[1] > np.shape(self.dot_size)[0]:
                    self.x_spacing = 10**5 / 10000000
                else:
                    self.x_spacing = 0.01 / 1e-07
            elif self.y_spacing is None and self.x_spacing is not None:
                if np.shape(self.dot_size)[0] > np.shape(self.dot_size)[1]:
                    self.y_spacing = 10**5 / 10000000
                else:
                    self.y_spacing = 0.01 / 1e-07
            else:
                self.x_spacing = 0.01
                self.y_spacing = 10**5
        self.xticks = np.linspace(0, self.x_spacing, self.ncols)
        self.yticks = np.linspace(0, self.y_spacing, self.nrows)
        self.x_pos = np.tile(self.xticks, self.nrows)
        self.y_pos = np.repeat(self.yticks, self.ncols)
        self.x_interval = self.xticks[1] - self.xticks[0]
        self.y_interval = self.yticks[1] - self.yticks[0]
        if run == True:
            self.make_dot_plot()

    def make_dot_plot(self):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_xlim(self.xticks[0] - self.x_interval/2, self.xticks[-1] +\
                         self.x_interval/2)
        self.ax.set_xticks(self.xticks)
        self.ax.set_yticks(self.yticks)
        self.ax.set_xticklabels(self.xticklabels,
                                rotation=self.x_rotation,
                                fontsize=self.fontsize)
        self.ax.set_yticklabels(self.yticklabels,
                                rotation=self.y_rotation,
                                fontsize=self.fontsize)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        sind = 0
        self._make_size_legend()
        dot_shape_provided = self._check_dot_shapes()
        color_provided = self._check_dot_colors()
        if color_provided == True:
            norm, mapper = self._make_colorbar()
        for norm, y in (zip(self.dot_size, self.yticks)):
            ind = 0
            for x, val in zip(self.x_pos, norm):
                if dot_shape_provided == False:
                    shape = self.shape
                else:
                    shape = self.dot_shape[sind, ind]
                if color_provided == False:
                    color = self.c
                    self.ax.scatter(x,
                                    y,
                                    s=val,
                                    c=color,
                                    edgecolor=self.edgecolor,
                                    lw=self.lw,
                                    marker=shape)
                else:
                    color = mapper.to_rgba(self.dot_color[sind, ind])
                    self.ax.scatter(x,
                                    y,
                                    s=val,
                                    color=color,
                                    edgecolor=self.edgecolor,
                                    lw=self.lw,
                                    marker=shape)
                ind += 1
            sind += 1
        if self.filename != '':
            self.fig.savefig(self.filename,
                             bbox_inches='tight',
                             transparent=True)

    def _normalize_dot_size(self, D):
        D = (D / np.max(D) * self.max_size).T
        return D

    def _check_dot_colors(self):
        if np.shape(self.dot_color) == np.shape(self.dot_size):
            color_provided = True
            if self.verbose == True:
                print('Dot colors provided')
        else:
            color_provided = False
            if self.verbose == True:
                print('Dot colors not provided')
        return color_provided

## Still need to work on implementing shape legend

    def _check_dot_shapes(self):
        if np.shape(self.dot_shape) == np.shape(self.dot_size):
            shape_provided = True
            if self.verbose == True:
                print('Dot shapes provided')
        else:
            shape_provided = False
            if self.verbose == True:
                print('Dot shapes not provided')
        return shape_provided

    def _make_size_legend(self):
        v = []
        labels = []
        value_ticks = np.linspace(np.min(self.dot_size_values),
                                  np.max(self.dot_size_values),
                                  self.legend_ticks)
        dot_ticks = np.linspace(np.min(self.dot_size), np.max(self.dot_size),
                                self.legend_ticks)
        for s, t in zip(dot_ticks, value_ticks):
            v.append(
                self.ax.scatter([], [],
                                s=s,
                                c='',
                                edgecolors=self.edgecolor,
                                lw=self.lw,
                                marker=self.shape))
            if self.round_point == 0:
                labels.append(int(t))
            else:
                labels.append(np.round(t, self.round_point))
        if self._check_dot_colors() == True:
            legend_pos = 1.40
        else:
            legend_pos = 1.01
        self.ax.legend(v,
                       labels,
                       ncol=1,
                       frameon=self.frameon,
                       title=self.legend_title,
                       borderpad=1,
                       scatterpoints=1,
                       loc=(legend_pos, 1),
                       bbox_to_anchor=(legend_pos, 0),
                       fontsize=self.ticksize)

    def _make_colorbar(self):
        minima = np.min(self.dot_color[self.dot_color > -np.inf])
        maxima = np.max(self.dot_color[self.dot_color < np.inf])
        norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=self.cmap)
        fmt = FormatScalarFormatter("%.2f")
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb1 = mpl.colorbar.ColorbarBase(cax,
                                        cmap=self.cmap,
                                        norm=norm,
                                        orientation='vertical',
                                        format=fmt)
        cb1.set_label(self.colorbar_title, labelpad=self.ticksize)
        cb1.ax.tick_params(labelsize=self.ticksize)
        if np.max(cb1.get_ticks()) > 100:
            cb1.formatter.set_powerlimits((0, 0))
        cb1.ax.yaxis.set_offset_position('left')
        cb1.update_ticks()
        if self.cbar_ticks == False:
            cb1.ax.set_yticklabels([])
        return (norm, mapper)


class volcano_plot():
    def __init__(self, df, run=True, min_fold=None, max_fold=None, \
                 min_p=None, max_p=None, cmap='viridis', filename='', **args):
        x = np.log2(df['fold'].values)
        y = -np.log10(df['adjusted-p'].values)
        self.df = df
        self._x = x
        self._y = y
        self.X = np.array([x, y]).T
        self.cmap = cmap
        self.filename = filename
        if min_fold:
            self._min_fold = min_fold
        else:
            self._min_fold = -np.inf
        if max_fold:
            self._max_fold = max_fold
        else:
            self._max_fold = np.inf
        if min_p:
            self._min_p = min_p
        else:
            self._min_p = -np.inf
        if max_p:
            self._max_p = max_p
        else:
            self._max_p = np.inf
        if run == True:
            self.run(**args)

    def run(self, **args):
        vector = np.ones(self.X.shape[0], dtype=bool)
        vector[np.abs(self._x) < self._min_fold] = False
        vector[np.abs(self._x) > self._max_fold] = False
        vector[self._y < self._min_p] = False
        vector[self._y > self._max_p] = False
        self.vector = vector
        plot = scatter(self.X,
                       run=False,
                       c=self.vector,
                       show_xticks=True,
                       show_yticks=True,
                       facecolor='w',
                       ylabel='-log10 adjusted-p',
                       xlabel='log2 fold',
                       cmap=self.cmap,
                       **args)
        fig, ax = plot.run()
        if self.vector.sum() > 0:
            for i in np.arange(self.vector.shape[0])[self.vector]:
                xy = self.X[i]
                if xy[0] < 0:
                    ha = 'right'
                else:
                    ha = 'left'
                ax.annotate(xy=xy, s=self.df.index.values[i], ha=ha)
        if self.filename != '':
            fig.savefig(self.filename, bbox_inches='tight', transparent=True)


def plot_by_wells(values,
                  well_position,
                  plate_format='384',
                  cmap='viridis',
                  figsize=(7, 5),
                  filename='',
                  operation='mean',
                  title=''):
    '''
    Make a heatmap that plots values on a 384 / 96 well plate.
    Assumes that columns are integer and rows are alphabets.
    '''
    X = convert_to_plate(values=values,
                         well_position=well_position,
                         plate_format=plate_format,
                         operation=operation)
    fig, ax = plt.subplots(figsize=figsize)
    imshow = ax.imshow(X=X, cmap=cmap)
    ax.set_yticks(np.arange(16))
    ax.set_yticklabels(X.index.values)
    ax.set_xticks(np.arange(24))
    ax.set_xticklabels(X.columns.values)
    ax.set_title(title, fontsize=12)
    ax.grid(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(imshow, cax=cax)
    if filename != '':
        fig.savefig(filename, bbox_inches='tight')


def convert_to_plate(values=None,
                     well_position=None,
                     plate_format='384',
                     operation='mean'):
    import string
    X = np.zeros((16, 24))
    if plate_format == '384':
        X = pd.DataFrame(0,
                         index=list(string.ascii_uppercase[:16]),
                         columns=np.arange(1, 25).astype(str))
    else:
        X = pd.DataFrame(0,
                         index=list(string.ascii_uppercase[:8]),
                         columns=np.arange(1, 13).astype(str))
    if values is not None and well_position is not None:
        rows = np.array([str(x[0]).upper() for x in well_position])
        cols = np.array([int(x[1:]) for x in well_position]).astype(str)
        well_pos = np.array([x + '_' + y for x, y in zip(rows, cols)])
        pos, counts = np.unique(well_pos, return_counts=True)
        for i in pos:
            subset_pos = (well_pos == i)
            coordinate = i.split('_')
            v = values[subset_pos]
            if operation == 'mean':
                v = np.mean(v)
            else:
                v = np.median(v)
            X.loc[coordinate[0], coordinate[1]] = v
    return X


class stratify():
    def __init__(self,
                 X,
                 labels,
                 c=None,
                 binsize=10,
                 figsize=(3, 3),
                 filename='',
                 plot=True,
                 ax=None,
                 ylabel='',
                 xlabel='',
                 order_cbar_labels=[]):
        assert binsize % 2 == 0, 'binsize must be an even number'
        self.binsize = binsize
        self.X = X
        self.figsize = figsize
        self.labels = labels
        self.c = c
        self.filename = filename
        self.plot = plot
        self.ax = ax
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.order_cbar_labels = order_cbar_labels
        self.bin_it()
        if self.plot == True:
            self.run()

    def bin_it(self, style='Percent'):
        length = np.shape(self.X)[0]
        df = pd.DataFrame(0,
                          index=np.unique(self.labels),
                          columns=range(length))
        for i in range(length):
            if i < int(
                    self.binsize / 2) or i > (length - int(self.binsize / 2)):
                if i < self.binsize:
                    l, n = 0, int(self.binsize / 2)
                else:
                    l, n = length - int(self.binsize / 2), length
            else:
                l, n = i - self.binsize, i + self.binsize
            lb, cnt = np.unique(self.labels[l:n], return_counts=True)
            df.loc[lb, i] = cnt
        self.df = df
        if style == 'Percent':
            self.df = self.df.div(self.df.sum(axis=0), axis=1) * 100
        elif style == 'Frequency':
            self.df = self.df.div(self.df.sum(axis=0), axis=1)
        if len(self.order_cbar_labels) > 0:
            self.df = self.df.loc[self.order_cbar_labels]

    def run(self):
        external_plot_control = True
        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            external_plot_control = False
        else:
            self.ax = self.ax.twinx()
        x = np.arange(1, np.shape(self.X)[0] + 1)
        if external_plot_control == True:
            self.ax.yaxis.tick_right()
        for i in self.df.index.values:
            if self.c is None:
                self.ax.plot(x, self.df.loc[i].values, label=i)
            else:
                self.ax.plot(x,
                             self.df.loc[i].values,
                             label=i,
                             color=self.c[i])
        self.ax.set_ylabel(self.ylabel, rotation=0)
        if self.xlabel != '':
            self.ax.set_xlabel(self.xlabel)
        if self.filename != '' and external_plot_control == False:
            self.fig.savefig(self.filename,
                             bbox_inches='tight',
                             transparent=True)
