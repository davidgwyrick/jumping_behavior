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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
def plot_state_usage(state_usage,K,ax=None,pdfdoc=None):
    ##===== Plot state usage =====## 
    if ax is None:
        fs=(8,8) if K < 11 else (16,8)
        fig,ax = plt.subplots(figsize=fs)
        plt.suptitle('State Usage for K={}'.format(K))

    sns.barplot(np.arange(K+1),state_usage*100,palette=color_palette,ax=ax)
    ax.set_xlabel('State')
    ax.set_ylabel('Usage',fontsize=12)

    tmp = [str(i) for i in np.arange(K)]
    tmp.append('NaN')
    ax.set_xticklabels(tmp)
    
    if pdfdoc is not None:
        pdfdoc.savefig(fig)
        plt.close(fig)
    
def plot_state_durations2(state_duration_list, state_usage, K, 
                         dsf=1,
                         SAVEFIG=False,
                         PlotDir='./plots',
                         fname=None):
    
    if SAVEFIG:
        if fname is None:
            'state-durations_K-{}.pdf'.format(K)
        #Create pdf to save all plots
        pdfdoc = PdfPages(os.path.join(PlotDir,fname))
    else:
        pdfdoc = None
           
    bSize = dsf/60
    nTrials = len(state_duration_list)
    for state in range(-1,K):
        s_dur = []
        for iTrial in range(nTrials):
            s_dur.extend(state_duration_list[iTrial][state])

        s_dur = np.asarray(s_dur)*bSize

        
        fig, ax = plt.subplots(figsize=(8,8))    
        ax.hist(s_dur,color=color_palette[state])
        ax.axvline(s_dur.mean(), color='k',ls='--', linewidth=2,zorder=2)
        ax.text(0.2, 0.5, 'n = {}'.format(len(s_dur)), transform=ax.transAxes,fontdict={'fontweight': 'normal', 'fontsize': 16})
        ax.text(0.2, 0.6, '\u03BC = {:.0f}ms'.format(s_dur.mean()*1E3), transform=ax.transAxes,fontdict={'fontweight': 'normal', 'fontsize': 16})
        
        if state == -1:
            ax.set_title('NaN State Durations')
        else:
            ax.set_title('State {} Durations'.format(state))

        ax.set_xlabel('Duration (sec)')

        if SAVEFIG:
            pdfdoc.savefig(fig)
            plt.close(fig)

    ##===== Plot state usage =====## 
    plot_state_usage(state_usage,K,pdfdoc=pdfdoc)
    
    if SAVEFIG:
        pdfdoc.savefig(fig)
        plt.close(fig)
    if SAVEFIG:
        pdfdoc.close()
        
def plot_state_durations(state_duration_list, state_usage, K, 
                         bSize=1/60,
                         SAVEFIG=False,
                         PlotDir='./plots',
                         fname=None):
    
    if SAVEFIG:
        if fname is None:
            'state-durations_K-{}.pdf'.format(K)
        #Create pdf to save all plots
        pdfdoc = PdfPages(os.path.join(PlotDir,fname))        
        
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
    
    if SAVEFIG:
        pdfdoc.savefig(fig)
        plt.close(fig)

    ##===== Plot state usage =====## 
    fs=(8,8) if K < 11 else (16,8)
    fig,ax = plt.subplots(figsize=fs)
    plt.suptitle('State Usage for K={}'.format(K))

    sns.barplot(np.arange(K+1),state_usage*100,palette=color_palette,ax=ax)
    ax.set_xlabel('State')
    ax.set_ylabel('Usage',fontsize=12)

    tmp = [str(i) for i in np.arange(K)]
    tmp.append('NaN')
    ax.set_xticklabels(tmp)
    
    if SAVEFIG:
        pdfdoc.savefig(fig)
        plt.close(fig)
    if SAVEFIG:
        pdfdoc.close()
        
def plot_xval_lls_vs_K(ll_training, ll_heldout,Ks, opt, SAVEFIG=False):

    fig, ax = plt.subplots(figsize=(8, 5))
#     ax.plot(Ks, ll_training[:,:-1], 'o',color=cc[0])#,label='training data kfolds')
#     ax.plot(Ks, ll_heldout[:,:-1], 'o',color=cc[1])#,label='heldout data kfolds')
#     ax.plot(Ks, ll_training[:,-1], 'Pk',label='Full Model Fit')
    ax.plot(Ks, np.mean(ll_training[:,:-1], axis=1), 'x-',color=cc[0], markersize=8,label='Mean across training kfolds')
    ax.plot(Ks, np.mean(ll_heldout[:,:-1], axis=1), 'x-',color=cc[1], markersize=8,label='Mean across heldout kfolds')    
    ax.set_ylabel('log-likelihood per time-step')
    ax.set_xlabel('# HMM states (K)')
    ax.set_xticks(Ks)
    ax.legend()

    ax.set_title('Cross-Validated Log-Likelihood')
    
    plt.savefig('./plots/ARHMM_xvalidation.png')
#     #Save Fig
#     if SAVEFIG:
# #        pdfdoc.savefig(fig)
# #        fig.savefig(pdfdoc)
#         tmpstr = '-'.join((opt['mID'],opt['condition'],opt['stimuli']))
#         fname = '{}_llhood-plot.png'.format(tmpstr)           
#         fpath = os.path.join(opt['SaveDirRoot'],fname)
#         fig.savefig(fpath)
#         plt.close('all')


def plot_dynamics_2d(arhmm,
                     mins=(-125,-125),
                     maxs=(125,125),
                     npts=35,
                     axis=None,
                     SAVEFIG=False,
                     PlotDir='./plots',
                     fname=None,
                     **kwargs):
    
    #Get model parameters
    K = arhmm.K
    dObs = arhmm.D
    
    if SAVEFIG:
        if fname is None:
            'AR-streamplots_K-{}.pdf'.format(arhmm.K)
        #Create pdf to save all plots
        pdfdoc = PdfPages(os.path.join(PlotDir,fname))        

    average_dx_dy = np.array([[-10.82328051,  10.58991895],[-12.93093803,  11.62390074]])
    ##===== Let's visualize the dynamics of each state  =====## 
    for iState in range(K):
        fig, axes = plt.subplots(1,3,figsize=(15,5))
        plt.suptitle('State {} Dynamics'.format(iState))

        #Get AR process parameters
        AR = arhmm.observations.As[iState]
        bias = arhmm.observations.bs[iState]
        xstar = np.matmul(np.linalg.inv(np.eye(dObs)-AR),bias)
        
        #Create grid of points
        x_grid, y_grid = np.meshgrid(np.linspace(mins[0], maxs[0], npts), np.linspace(mins[1], maxs[1], npts))
        nX,nY = x_grid.shape
        xy_grid_nose = np.column_stack((x_grid.ravel(), y_grid.ravel(), np.zeros((npts**2,0))))
        #Using average distance between nose and eye, and eye and ear, create separate grids for those points
        xy_grid_leye = xy_grid_nose - average_dx_dy[0,:]
        xy_grid_lear = xy_grid_leye - average_dx_dy[1,:]
        xy_grid_3pts = np.hstack((xy_grid_nose,xy_grid_leye,xy_grid_lear))

        #Create delta grid
        dx = xy_grid_3pts.dot(AR.T) + bias - xy_grid_3pts
        dx = dx.reshape((nX,nY,-1))

        for ii, ptstr in enumerate(['Nose','LEye','LEar']):
            ax = axes[ii]

            #Calculate speed for linewidth
            speed = np.sqrt(dx[:,:,ii*2]**2 + dx[:,:,ii*2+1]**2)
            #Better than a quiver plot
            ax.streamplot(xy_grid_3pts[:,ii*2].reshape((nX,nY)), xy_grid_3pts[:,ii*2+1].reshape((nX,nY)), dx[:,:,ii*2], dx[:,:,ii*2+1],
                          color=cc[iState],arrowsize=1.25,arrowstyle='->',linewidth= 5*speed / speed.max())
            
#             ax.plot(xstar[2*ii],xstar[2*ii+1],'*k',ms=15)
            ax.set_title('Side {}'.format(ptstr))
            ax.set_xlim([-100,100]); ax.set_ylim([-100,100])
            ax.plot(0,0,'+k',ms=15)
            
        if SAVEFIG:
            pdfdoc.savefig(fig)
            plt.close(fig)
    if SAVEFIG:
        pdfdoc.close()
        
        
def plot_eigenvalues(A):
    
    w,v = np.linalg.eig(A)
    xstar = np.matmul(np.linalg.inv(np.eye(dObs)-A),b)
            
    #Plot eigenvalues of A
    ax = axes[0]
    ax.plot(np.real(w),np.imag(w),'.k',MarkerSize=9)
    axlims = ax.axis()
    ax.add_patch(plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None',lw=2))
    ax.axis(axlims)        
    ax.set_xlabel('Real'); ax.set_ylabel('Imag')
    ax.set_title('Eigenvalues of A'); ax.grid()
            
def plot_AR_matrices(arhmm,
                     dsf=1,
                     SAVEFIG=False,
                     PlotDir='./plots',
                     fname=None,
                     **kwargs):
    
    #Get model parameters
    K = arhmm.K
    dObs = arhmm.D
    AR_lags = arhmm.observations.lags
    samp_rate = 60/dsf
    if SAVEFIG:
        if fname is None:
            'AR-matrices_-{}.pdf'.format(K)
        #Create pdf to save all plots
        pdfdoc = PdfPages(os.path.join(PlotDir,fname))        

    for iState in range(K):
        A = arhmm.observations.As[iState]
        b = arhmm.observations.bs[iState].reshape(-1,1)

        fig, axes = plt.subplots(1,2,figsize=(5*AR_lags+1,5),gridspec_kw = {'width_ratios':[5*AR_lags,1]})
  
        #Plot actual A matrix
        AmI = (A-np.tile(np.eye(dObs),AR_lags))*samp_rate
        vm = np.max(np.abs(AmI))
        sns.heatmap(AmI,ax=axes[0],vmin = -1*vm, vmax=vm,annot=True,cmap='RdBu_r',square=True,cbar_kws={'shrink':0.5})
        axes[0].set_title('(A-I)/dt Matrix')

        #Plot fixed point
        if AR_lags == 1:
            w,v = np.linalg.eig(A)
            xstar = np.matmul(np.linalg.inv(np.eye(dObs)-A),b)
            axes[1].set_title('x*')
            sns.heatmap(xstar,ax=axes[1],annot=True,square=True,cbar_kws={'shrink':0.5})
        else:
            axes[1].set_title('bias')
            sns.heatmap(b,ax=axes[1],annot=True,square=True,cbar_kws={'shrink':0.5})
            
        if SAVEFIG:
            pdfdoc.savefig(fig)
            plt.close(fig)
    if SAVEFIG:
        pdfdoc.close()
        
def plot_example_trajectories(state_duration_list,
                              state_startend_list,
                              mean_state_durations,
                              data_list, arhmm,
                              dsf=1,
                              longest=True,
                              simulated=False,
                              nTraces = 75,
                              SAVEFIG=False,
                              PlotDir='./plots',
                              fname=None):
    np.random.seed(0)
    #Get parameters
    K = arhmm.K; dObs=arhmm.D
    AR_lags = arhmm.observations.lags
    nTrials = len(data_list)
    bSize = dsf/60
    
    # Calculate the max state duration for each state on each trial
    state_durations_max = np.zeros((nTrials,K))
    for iTrial, sd_trial in enumerate(state_duration_list):
        for iState, sd in enumerate(sd_trial[:-1]):
            if len(sd) == 0:
                state_durations_max[iTrial,iState] = 0
            else:
                if longest:
                    state_durations_max[iTrial,iState] = np.max(sd)
                else:
                    state_durations_max[iTrial,iState] = np.median(sd)
     
    if SAVEFIG:
        if fname is None:
            if simulated:
                'state-trajectories_simulated_K-{}.pdf'.format(K)
            else:
                'state-trajectories_data_K-{}.pdf'.format(K)
        #Create pdf to save all plots
        pdfdoc = PdfPages(os.path.join(PlotDir,fname))        

    for iState in range(K):
        #Get trial indices sorted by state duration for state
#         trial_indices = np.argsort(state_durations_max[:,iState],axis=0)[::-1]
        if longest:
            trial_indices = np.argsort(state_durations_max[:,iState],axis=0)[::-1]
        else:
        #Or get random trial indices
            trial_indices = np.random.randint(0,nTrials,nTrials)
            
        fig, axes = plt.subplots(1,3,figsize=(15,5))
        nPlot = 0; trial_lengths = []
        #Loop over trials and plot actual traces
        for iTrial in trial_indices:
            if nPlot > nTraces:
                break
                
            #Get the correct index
            if longest:
                indy = np.argmax(state_duration_list[iTrial][iState])
                nT = np.max(state_duration_list[iTrial][iState])
            else:
                #Randomly choose a segment from this trial
                nSegs = len(state_duration_list[iTrial][iState])
                if nSegs == 0:
                    continue
                indy = np.random.randint(0,nSegs)
                nT = state_duration_list[iTrial][iState][indy]
                
                #But only take it if the segment is longer than average state duration
                if (nT < mean_state_durations[iState]) | (nT > mean_state_durations[iState] + 12):
                    continue
                        
            iS = state_startend_list[iTrial][iState][indy,0]
            iE = state_startend_list[iTrial][iState][indy,1]-1

            trial_lengths.append(nT)
            nPlot += 1
            
            #Get data
            if simulated:
                #Initialize simlation based on the data
                data = np.zeros((nT,dObs))
                for iT in range(AR_lags):
                    data[iT,:] = data_list[iTrial][iS+iT,:]
          
                #Evolve trajectory based on model parameters
                tmp_stateseq = np.repeat(iState,nT); pad = 0
                for iT in range(AR_lags,nT):
                    data[iT,:] = arhmm.observations.sample_x(tmp_stateseq[iT], data[:iT,:],input = np.zeros((nT+pad,)), with_noise=False)
            else:
                data = data_list[iTrial][slice(iS,iE),:]     

            #Plot trajectories for each point
            for iPt, ptstr in enumerate(['Nose','LEye','LEar']):
                ax = axes[iPt]
                ax.plot(data[:,2*iPt],data[:,2*iPt+1],'-k',alpha=0.5)

                ax.plot(data[0,2*iPt],data[0,2*iPt+1],'s',color='w',markersize=6,markeredgecolor='k',markeredgewidth=1)
                ax.plot(data[-1,2*iPt],data[-1,2*iPt+1],'o',color=cc[iState],markersize=6,markeredgecolor='k',markeredgewidth=1)
                ax.plot(0,0,'+k',ms=15)
                ax.set_xlim([-100,100]); ax.set_ylim([-100,100])
                
        if simulated:
            plt.suptitle('{} simulated traces for State {}: \u03BC = {:.1f}ms  '.format(nPlot,iState,np.mean(trial_lengths)*1E3*bSize))
        else:
            plt.suptitle('{} traces for State {}: \u03BC = {:.1f}ms '.format(nPlot,iState,mean_state_durations[iState]*1E3*bSize)) 
        if SAVEFIG:
            pdfdoc.savefig(fig)
            plt.close(fig)
    if SAVEFIG:
        pdfdoc.close()
        
        
def plot_example_state_sequences(trMAPs, trMasks, data_list, K,
                                 nTrials_plot = 100,
                                 SAVEFIG=False,
                                 PlotDir='./plots',
                                 fname=None):
    
     
    if SAVEFIG:
        if fname is None:
            'state-sequences_ARHMM_K-{}.pdf'.format(arhmm.K)
        #Create pdf to save all plots
        pdfdoc = PdfPages(os.path.join(PlotDir,fname))        
    
    for iTrial in range(nTrials_plot):
        mapseq = trMAPs[iTrial]
        mask = trMasks[iTrial]
        data = data_list[iTrial]

        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_title('Trial {} Nose trajectory'.format(iTrial))
        ax.plot(data_list[iTrial][:,0],data_list[iTrial][:,1],'-k',alpha=0.5)
        # ax.plot(data_list[iTrial][:,2],data_list[iTrial][:,3],'-',color=usrplt.cc[-1],alpha=0.5)
        # ax.plot(data_list[iTrial][:,4],data_list[iTrial][:,5],'-',color=usrplt.cc[-1],alpha=0.5)

        for state in np.unique(mapseq):
            indy = np.where(mapseq == state)[0]
            #Plot traces
            ax.plot(data[indy,0],data[indy,1],'.',color=cc[state],ms=10)
        #     ax.plot(data[indy,2],data[indy,3],'.',color=usrplt.cc[state],ms=10)
        #     ax.plot(data[indy,4],data[indy,5],'.',color=usrplt.cc[state],ms=10)
        indy = np.where(~mask)[0]
        ax.plot(data[indy,0],data[indy,1],'.k',ms=10,label='NaN State')
        ax.legend()
        ax.plot(0,0,'+k',ms=20)
        ax.set_aspect('equal')
        if SAVEFIG:
            pdfdoc.savefig(fig)
            plt.close(fig)
    if SAVEFIG:
        pdfdoc.close()
        
def plot_transition_probability_matrix(TPM, pdfdoc=None):
    K = TPM.shape[-1]
    #Create masks for diagonal
    diag_mask = np.ones((K,K),dtype=bool)
    diag_mask[np.diag_indices(K)] = False
    
    fig, ax = plt.subplots(figsize=(8,8))
    #Plot annoted probabilites for diagonal, but based colormap on off-diagonals
    sns.heatmap(TPM,ax=ax,mask=diag_mask,square=True,annot=True,fmt='.2f',cbar=False,
                cmap='rocket_r',vmax=np.max(TPM[diag_mask]),vmin=np.max(TPM[diag_mask]))
    sns.heatmap(TPM,ax=ax,mask=~diag_mask,annot=False,square=True,cbar_kws={'shrink':0.5})

    ax.set_title('Transition Probability Matrix')
    ax.set_xlabel('State j'); ax.set_ylabel('State i')

    if pdfdoc is not None:
        pdfdoc.savefig(fig)
        plt.close(fig)
        
def plot_transition_matrices_per_condition(TCMs, data_df,
                                           TPM=None,
                                           SAVEFIG=False,
                                           PlotDir='./plots',
                                           fname=None):
    K = TCMs.shape[-1]
    conditions = np.unique(data_df['condition'])
    distances = np.unique(data_df['distance'])

    if SAVEFIG:
        if fname is None:
            'transition-count-matrices_K-{}.pdf'.format(K)
        #Create pdf to save all plots
        pdfdoc = PdfPages(os.path.join(PlotDir,fname))      
    
    #Create masks for diagonal
    diag_mask = np.ones((K,K),dtype=bool)
    diag_mask[np.diag_indices(K)] = False
        
    #Plot the transition probability matrix from the model
    if TPM is not None:
        plot_transition_probability_matrix(TPM,pdfdoc)

    #Get trial indices 
    indy_dict = data_df.groupby(['condition','distance','success']).indices

    #Loop over each condition
    for cond in conditions:

        fig, axes = plt.subplots(3,6,figsize=(20.25,12),gridspec_kw={'width_ratios':[4,4,4,4,4,0.25],'wspace':0.5,'hspace':0.25})
        plt.suptitle('All Mice, {} condition -> empirical transition matrices'.format(cond),y=0.95)

        #Calculate the max value for the colormap
        trial_indices = np.where(data_df['condition'] == cond)[0]
        TCM_uniq = np.sum(TCMs[trial_indices],axis=0)
        TCM_uniq = TCM_uniq/np.sum(TCM_uniq)
        vm = np.max(TCM_uniq[diag_mask])
        
        diff_list = []
        #Loop over each distance
        for iD,dist in enumerate(distances):
            tmp_list = []
            
            #And outcome
            for success in [0,1]:
                if (cond,dist,success) not in indy_dict.keys():
                    ax = axes[success,iD]
                    ax.set_title('{} / {} / {}'.format(success,dist,len(trial_indices)))
                    ax.axis('off')
                    continue
                #Get indices for this unique trial combination
                trial_indices = indy_dict[(cond,dist,success)]
                
                #Sum all transitions from all mice
                TCM_uniq = np.sum(TCMs[trial_indices],axis=0)
                #Normalize based on # of bins
                TCM_uniq = TCM_uniq/np.sum(TCM_uniq)
                #Save to calculate difference
                tmp_list.append(np.copy(TCM_uniq))
                #Mask out diagonal
                TCM_uniq[np.diag_indices(K)] = 0

                #Plot empirical transition matrix
                ax = axes[success,iD]
                if iD == 4:
                    sns.heatmap(TCM_uniq,ax=ax,cbar=True,vmin=0,vmax=vm,square=True,cbar_ax=axes[success,-1],cbar_kws={'shrink':0.5})
                else:
                    sns.heatmap(TCM_uniq,ax=ax,cbar=False,vmin=0,vmax=vm,square=True)

                ax.set_title('S:{} / D:{} / N:{}'.format(success,dist,len(trial_indices)))
                ax.set_xlabel('State j'); ax.set_ylabel('State i')

            if len(tmp_list) < 2:
                ax = axes[2,iD]
                ax.axis('off')
                diff_list.append([])
                continue
            else:
                diff_list.append(tmp_list[1]-tmp_list[0])

        #Calculate min/max 
        vm = np.max(np.abs(np.concatenate(diff_list))[:])
        
        #Loop over each distance
        for iD,dist in enumerate(distances):
            if len(diff_list[iD]) == 0:
                continue
                
            ax = axes[2,iD]
            if iD == 4:
                sns.heatmap(diff_list[iD],ax=ax,cbar=True,cmap='RdBu_r',vmin=-1*vm,vmax=vm,square=True,cbar_ax=axes[2,-1],cbar_kws={'shrink':0.5})
            else:
                sns.heatmap(diff_list[iD],ax=ax,cbar=False,cmap='RdBu_r',vmin=-1*vm,vmax=vm,square=True)
            ax.set_title('Success - Failure')
            ax.set_xlabel('State j'); ax.set_ylabel('State i') 

        pdfdoc.savefig(fig)
        plt.close(fig)

    pdfdoc.close()
    
def plot_transition_matrices_per_mouse(TCMs, data_df,
                                       TPM=None,
                                       SAVEFIG=False,
                                       PlotDir='./plots',
                                       fname=None):
    K = TCMs.shape[-1]
    mouseIDs = np.unique(data_df['subject'])
    conditions = np.unique(data_df['condition'])
    distances = np.unique(data_df['distance'])

    if SAVEFIG:
        if fname is None:
            'transition-count-matrices_K-{}.pdf'.format(K)
        #Create pdf to save all plots
        pdfdoc = PdfPages(os.path.join(PlotDir,fname))      
    
    #Create masks for diagonal
    diag_mask = np.ones((K,K),dtype=bool)
    diag_mask[np.diag_indices(K)] = False
        
    #Plot the transition probability matrix from the model
    if TPM is not None:
        plot_transition_probability_matrix(TPM,pdfdoc)

    #Get trial indices 
    indy_dict = data_df.groupby(['subject','condition','distance','success']).indices

    #Loop over each mouse, since mice may have different behaviors
    for mID in mouseIDs:
        
        #Loop over each condition
        for cond in conditions:

            fig, axes = plt.subplots(3,6,figsize=(20.25,12),gridspec_kw={'width_ratios':[4,4,4,4,4,0.25],'wspace':0.5,'hspace':0.25})
            plt.suptitle('Mouse {}, {} condition -> empirical transition matrices'.format(mID,cond),y=0.95)

            #Calculate the max value for the colormap
            trial_indices = np.where((data_df['condition'] == cond) & (data_df['subject'] == mID))[0]
            TCM_uniq = np.sum(TCMs[trial_indices],axis=0)
            TCM_uniq = TCM_uniq/np.sum(TCM_uniq)
            vm = np.max(TCM_uniq[diag_mask])

            diff_list = []
            #Loop over each distance
            for iD,dist in enumerate(distances):
                tmp_list = []

                #And outcome
                for success in [0,1]:
                    if (mID,cond,dist,success) not in indy_dict.keys():
                        ax = axes[success,iD]
                        ax.set_title('S:{} / D:{} / N:0'.format(success,dist))
                        ax.axis('off')
                        continue
                    #Get indices for this unique trial combination
                    trial_indices = indy_dict[(mID,cond,dist,success)]

                    #Sum all transitions from all mice
                    TCM_uniq = np.sum(TCMs[trial_indices],axis=0)
                    #Normalize based on # of bins
                    TCM_uniq = TCM_uniq/np.sum(TCM_uniq)
                    #Save to calculate difference
                    tmp_list.append(np.copy(TCM_uniq))
                    #Mask out diagonal
                    TCM_uniq[np.diag_indices(K)] = 0

                    #Plot empirical transition matrix
                    ax = axes[success,iD]
                    if iD == 4:
                        sns.heatmap(TCM_uniq,ax=ax,cbar=True,vmin=0,vmax=vm,square=True,cbar_ax=axes[success,-1],cbar_kws={'shrink':0.5})
                    else:
                        sns.heatmap(TCM_uniq,ax=ax,cbar=False,vmin=0,vmax=vm,square=True)

                    ax.set_title('S:{} / D:{} / N:{}'.format(success,dist,len(trial_indices)))
                    ax.set_xlabel('State j'); ax.set_ylabel('State i')

                if len(tmp_list) < 2:
                    ax = axes[2,iD]
                    ax.axis('off')
                    diff_list.append(np.zeros((K,K)))
                    continue
                else:
                    diff_list.append(tmp_list[1]-tmp_list[0])

            #Calculate min/max 
            vm = np.max(np.abs(np.concatenate(diff_list))[:])

            #Loop over each distance
            for iD,dist in enumerate(distances):
                if np.count_nonzero(diff_list[iD]) == 0:
                    continue

                ax = axes[2,iD]
                if iD == 4:
                    sns.heatmap(diff_list[iD],ax=ax,cbar=True,cmap='RdBu_r',vmin=-1*vm,vmax=vm,square=True,cbar_ax=axes[2,-1],cbar_kws={'shrink':0.5})
                else:
                    sns.heatmap(diff_list[iD],ax=ax,cbar=False,cmap='RdBu_r',vmin=-1*vm,vmax=vm,square=True)
                ax.set_title('Success - Failure')
                ax.set_xlabel('State j'); ax.set_ylabel('State i') 

            pdfdoc.savefig(fig)
            plt.close(fig)

    pdfdoc.close()       