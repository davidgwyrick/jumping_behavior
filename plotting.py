#Misc
import time, os, sys, pdb
from glob import glob
from fnmatch import fnmatch

#Base
import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.io as sio
import scipy.stats as st

#Save
import json
import scipy.io as sio
import h5py

#Plot
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as colors

#User
import util

color_names=['windows blue','red','amber','faded green','dusty purple','orange','steel blue','pink','mint',
             'clay','light cyan','forest green','pastel purple','salmon','dark brown','lavender','pale green',
             'dark red','gold','dark teal','rust','fuchsia','pale orange','cobalt blue','mahogany','cloudy blue',
             'dark pastel green','dust','electric lime','fresh green','light eggplant','nasty green','greyish']
 
color_palette = sns.xkcd_palette(color_names)
cc = sns.xkcd_palette(color_names)

#Default Plotting Options
sns.set_style("ticks")

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    
def gradient_cmap(gcolors, nsteps=256, bounds=None):
    from matplotlib.colors import LinearSegmentedColormap
    ncolors = len(gcolors)
    if bounds is None:
        bounds = np.linspace(0, 1, ncolors)

    reds = []
    greens = []
    blues = []
    alphas = []
    for b, c in zip(bounds, gcolors):
        reds.append((b, c[0], c[0]))
        greens.append((b, c[1], c[1]))
        blues.append((b, c[2], c[2]))
        alphas.append((b, c[3], c[3]) if len(c) == 4 else (b, 1., 1.))

    cdict = {'red': tuple(reds),
             'green': tuple(greens),
             'blue': tuple(blues),
             'alpha': tuple(alphas)}

    cmap = LinearSegmentedColormap('grad_colormap', cdict, nsteps)
    return cmap

def plot_z_samples(zs,mask,used_states,title=None,ax=None,pdf=None,apply_mask=True):
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

    # How many states were discovered?
    K = len(used_states)
    #Make appropriate colormap
    cc = sns.xkcd_palette(color_names[:K])
    
    if apply_mask:
        Zm = np.ma.masked_where(mask,zs)
    else:
        Zm = zs

    # Plot StateSeq as a heatmap
    im = ax.imshow(Zm, aspect='auto', vmin=0, vmax=K - 1,cmap=gradient_cmap(cc), interpolation="nearest")

    # Create a legend
    plt.legend(used_states, loc=4)

    ax.set_ylabel("Trial")
    ax.set_xlabel("Time (10ms bins)")
    if title is not None:
        ax.set_title(title)

    if pdf is not None:
        pdf.savefig(fig)
        plt.close('all')
    

def plot_MAP_estimates(trMAPs,trMASKs,used_states,trsum,session=1,apply_mask=True):
    # Get number of trials from trial summary data frame
    nTrials = len(trsum)
    timestr = time.strftime('%Y%m%d_%H%M')
   
    # Plot the discrete state sequence of the collection of trials
#     title = 'MAP estimates for all trials of session {}'.format(session)
#     plot_z_samples(trMAPs,used_states,title=title) 

    # Loop over conditions
    for cond in ['pre','post']:
        # Loop over locomotion conditions
        for lr in Run_Dict:
            # Find the indices of the trlist that correspond to the condition
            indy = np.where((trsum['run'] == lr) & (trsum['cond'] == cond))[0]

            # Create a new list based of only that condition
            cond_MAPs = np.array([trMAPs[iTrial] for iTrial in indy])
            state_Masks = np.array([trMASKs[iTrial] for iTrial in indy])
            
            # Plot the discrete state sequence of the collection of trials
            title = '{}-DOI condition / {} State -> {} trials'.format(cond.capitalize(),Run_Dict[lr],len(cond_MAPs))
            plot_z_samples(cond_MAPs,state_Masks,used_states,title = title)

    # Close PDF file
    # pdfdoc.close()
    
def plot_state_durations(state_duration_list, K, bSize=1/60):
    
    nRows = 2; nCols = int(K/2)
    #Create figure
    fig,axes = plt.subplots(nRows,nCols+1,figsize=(15,5),gridspec_kw={'wspace':0.5,'hspace':0.5})
    plt.suptitle('Jumping Behavior State Durations')

    nTrials = len(state_duration_list)
    for state in range(-1,K):
        s_dur = []
        for iTrial in range(nTrials):
            s_dur.extend(state_duration_list[iTrial][state])

        s_dur = np.asarray(s_dur)*bSize
        if state == -1:
            ax = axes[1,-1]
        else:
            ax = axes[state//nCols,state%nCols]

        ax.hist(s_dur,color=color_palette[state])
        ax.axvline(s_dur.mean(), color=color_palette[state], linewidth=2)
        ax.text(0.2, 0.5, 'n = {}'.format(len(s_dur)), transform=ax.transAxes,fontdict={'fontweight': 'normal', 'fontsize': 10})
        ax.text(0.2, 0.6, '\u03BC = {:.0f}ms'.format(s_dur.mean()/0.001), transform=ax.transAxes,fontdict={'fontweight': 'normal', 'fontsize': 10})
        
        if state == -1:
            ax.set_title('NaN State')
        else:
            ax.set_title('State {}'.format(state))
        if state//nCols == 1:
            ax.set_xlabel('Duration (sec)')
        else:
            ax.set_xlabel('')
        ax.tick_params(axis="y",direction="in", pad=-22)



def plot_state_usages(state_usage_list, trsum,interval=None):
    #How many states were used? -1 for NaN state
    K = len(state_usage_list[0])-1
    nT = np.sum(state_usage_list[0])
    
    fig,axes = plt.subplots(2,2,figsize=(16,16))
    if interval is None:
        plt.suptitle('Overall State Usages',y=0.925,fontsize=18)
    else:
        plt.suptitle('State Usages over the {}:{} interval'.format(*interval),y=0.925,fontsize=18)
        
    Run_Dict = {0:'Rest',1:'Running'}
    nOri = 12; nSpf = 6
    for ii,cond in enumerate(['pre','post']):
        # Separate by locomotion conditions
        for rr in Run_Dict:
            # Find the indices of the trlist that correspond to the condition
            indy = np.where((trsum['run'] == rr) & (trsum['cond'] == cond))[0]
            cond_su = np.array([state_usage_list[iTrial] for iTrial in indy])/nT
    #         cond_su = np.mean([state_usage_list[iTrial] for iTrial in indy],axis=0)/nT

            # Plot overall state_usages
            ax = axes[ii,rr]
            ax.set_xticks(np.arange(K))
            ax.set_xticklabels(np.arange(K))

    #         ax.bar(np.arange(K+1), cond_su, color=dplot.color_palette)
            sns.barplot(data = cond_su ,orient='v',ci = 95, ax=ax,palette=color_palette)
            ax.set_xticklabels([*list(map(str,np.arange(K))),'NaN'])
            ax.set_title('{}-DOI, {} -> {} trials'.format(cond,Run_Dict[rr],len(indy)))        
            ax.set_ylabel('Usage')
            ax.set_xlabel('State')

    # Loop over locomotion conditions
    for rr in Run_Dict:
        fig,axes = plt.subplots(2,nOri,figsize=(28,7))
        if interval is None:
            plt.suptitle('State Usages for {} Behavioral Condition'.format(Run_Dict[rr]),y=0.975,fontsize=18)
        else:
            plt.suptitle('State Usages for {} Behavioral Condition over the {}:{} interval'.format(Run_Dict[rr],*interval),y=0.975,fontsize=18)

        for ii,cond in enumerate(['pre','post']):
            for iOri in range(nOri):
                # Find the indices of the trlist that correspond to the condition
                indy = np.where((trsum['run'] == rr) & (trsum['cond'] == cond) & (trsum['ori'] == iOri+1))[0]
                cond_su = np.array([state_usage_list[iTrial] for iTrial in indy])/nT
                if len(indy) < 1:
                    continue
                # Plot overall state_usages
                ax = axes[ii,iOri]
                sns.barplot(data = cond_su[:,:-1] ,orient='v', ax=ax,ci = 95,palette=color_palette)
                if ii == 1:
                    ax.set_xticklabels([*list(map(str,np.arange(K)))]) #,'NaN'
                    ax.set_xlabel('State')
                else:
                    ax.set_xticks([])
                ax.set_title('{}\u00B0 -> {} trials'.format(iOri*30,len(indy))) 
                ax.set_ylim([0,0.5])
                if iOri == 0:
                    ax.set_ylabel('{}-DOI Usage'.format(cond))
                else:
                    ax.set_yticklabels([])    
    
def plot_xval_lls_vs_K(ll_training, ll_heldout,Ks, opt, SAVEFIG=False):

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Ks, ll_training[:,:-1], 'o',color=cc[0])#,label='training data kfolds')
    ax.plot(Ks, ll_heldout[:,:-1], 'o',color=cc[1])#,label='heldout data kfolds')
    ax.plot(Ks, ll_training[:,-1], 'Pk',label='Full Model Fit')
    ax.plot(Ks, np.mean(ll_training[:,:-1], axis=1), 'x-',color=cc[0], markersize=8,label='Mean across training kfolds')
    ax.plot(Ks, np.mean(ll_heldout[:,:-1], axis=1), 'x-',color=cc[1], markersize=8,label='Mean across heldout kfolds')    
    ax.set_ylabel('log-likelihood per time-step')
    ax.set_xlabel('# HMM states (K)')
    ax.legend()

    ax.set_title('Cross-Validated Log-Likelihood')
        
    #Save Fig
    if SAVEFIG:
#        pdfdoc.savefig(fig)
#        fig.savefig(pdfdoc)
        tmpstr = '-'.join((opt['mID'],opt['condition'],opt['stimuli']))
        fname = '{}_llhood-plot.png'.format(tmpstr)           
        fpath = os.path.join(opt['SaveDirRoot'],fname)
        fig.savefig(fpath)
        plt.close('all')
    else:
        plt.show()
    

def plot_dynamics_2d(dynamics_matrix,
                     bias_vector,
                     mins=(-40,-40),
                     maxs=(40,40),
                     npts=20,
                     axis=None,
                     **kwargs):
    """Utility to visualize the dynamics for a 2 dimensional dynamical system.

    Args
    ----

        dynamics_matrix: 2x2 numpy array. "A" matrix for the system.
        bias_vector: "b" vector for the system. Has size (2,).
        mins: Tuple of minimums for the quiver plot.
        maxs: Tuple of maximums for the quiver plot.
        npts: Number of arrows to show.
        axis: Axis to use for plotting. Defaults to None, and returns a new axis.
        kwargs: keyword args passed to plt.quiver.

    Returns
    -------

        q: quiver object returned by pyplot
    """
    assert dynamics_matrix.shape == (2, 2), "Must pass a 2 x 2 dynamics matrix to visualize."
    assert len(bias_vector) == 2, "Bias vector must have length 2."

    x_grid, y_grid = np.meshgrid(np.linspace(mins[0], maxs[0], npts), np.linspace(mins[1], maxs[1], npts))
    xy_grid = np.column_stack((x_grid.ravel(), y_grid.ravel(), np.zeros((npts**2,0))))
    dx = xy_grid.dot(dynamics_matrix.T) + bias_vector - xy_grid

    if axis is not None:
        q = axis.quiver(x_grid, y_grid, dx[:, 0], dx[:, 1], **kwargs)
    else:
        q = plt.quiver(x_grid, y_grid, dx[:, 0], dx[:, 1], **kwargs)
    return q