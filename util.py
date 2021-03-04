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
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
ResultsDir = '/home/dwyrick/projects/jumping_behavior/results'

def read_data(fpath='./data/jumping_data_102220.h5', dsf=1):
    
    ##===== Read in Data =====##
    data_df = pd.read_hdf(fpath)
    nTrials = len(data_df)
    data_df.head()

    #DLC tracking confidence threshold at which to mask out data
    confidence_threshold = 0.8

    #Loop over trials and reformat data for ARHMM
    data_list = []; mask_list = []
    for iTrial in range(nTrials):
        #Get coordinates of Take-Off platform
        xc = np.nanmean(data_df.loc[iTrial]['Side TakeFL x'])
        yc = np.nanmean(data_df.loc[iTrial]['Side TakeFL y'])

        xy_list = []; ll_list = []
        for ii, ptstr in enumerate(['Nose','LEye','LEar']):
            x = data_df.loc[iTrial]['Side {} x'.format(ptstr)]
            y = data_df.loc[iTrial]['Side {} y'.format(ptstr)]
            llhood = data_df.loc[iTrial]['Side {} likelihood'.format(ptstr)]

            #Coordinates relative to take-off platform
            xy_list.append((x-xc,y-yc))

            #Create mask for points that have a confidence lower than the given threshold
            mask = llhood > confidence_threshold
            ll_list.append((mask,mask))

        #Downsample if required #data_list_fullres.append(tmp)
        tmp = np.vstack(xy_list)[:,2:-2].T; tmp2 = tmp if dsf == 1 else tmp[:-1:dsf,:]; data_list.append(tmp2)
        tmp = np.vstack(ll_list)[:,2:-2].T; tmp2 = tmp if dsf == 1 else tmp[:-1:dsf,:]; mask_list.append(tmp2)
        
    return data_list, mask_list, data_df
    
def params_to_dict(params, opt, AR_INPUTS=False):
    """
    Intended usage: 
    When setting up model, define
        model_type = dict(HMM_INPUTS=False, HMM_RECURRENCE=False, R_INPUTS=False, ROBUST=True)
    etc, then run 
    par_dict = params_to_dict(arhmm.params, model_type)
    and save par_dict to file using io_dict_to_hdf5.py.
    """
    init = {}
    init['P0'] = params[0][0]
    
    trans = {}
    trans['log_Ps'] = params[1][0]

    if opt['transitions'] == 'inputdriven':
        trans['Ws'] = params[1][1]
    elif opt['transitions'] == 'recurrent':
        trans['Rs'] = params[1][1]
    
    obs = {}    
    obs['As'] = params[2][0]
    obs['bs'] = params[2][1]
    obs['ABs'] = np.concatenate((obs['As'],obs['bs'][:,:,None]),axis=-1)
    
    obs['sqrt_Sigmas'] = params[2][3]
    
    if AR_INPUTS:    
        obs['Vs'] = params[2][2]
    
    if opt['observations'] == 'robust_autoregressive':  
        obs['log_nus'] = params[2][4]
        obs['nus'] = np.exp(params[2][4])
    
    out = {}
    out['init_state_distn'] = init
    out['transitions'] = trans
    out['observations'] = obs  
    
    return out
    
def dict_to_params(par_dict):
    params = [[] for ii in range(3)]
    params[0].append( par_dict['init_state_distn']['P0'] )
    
    params[1].append( par_dict['transitions']['log_Ps'] )
    if 'Ws' in par_dict['transitions'].keys():
        params[1].append( par_dict['transitions']['Ws'] )
    if 'Rs' in par_dict['transitions'].keys():
        params[1].append( par_dict['transitions']['Rs'] )
        
    params[2].append( par_dict['observations']['As'] )
    params[2].append( par_dict['observations']['Bs'] )  
    if 'Vs' in par_dict['observations'].keys():
        params[2].append( par_dict['observations']['Vs'] ) 
    else:
        params[2].append( np.zeros((len(par_dict['observations']['Bs']), 0)) )      #empty array of correct size
    params[2].append( par_dict['observations']['sqrt_Sigmas'] )
    if 'log_nus' in par_dict['observations'].keys():
        params[2].append( par_dict['observations']['log_nus'] ) 
                         
    return params

def make_base_dir(opt, SAVE=True):
    #Create base folder for results    
    timestr = time.strftime('%Y-%m-%d_%H%M')

#     fit_str = '{}-ARHMM_lag-{}_K-{}_dsf-{}_{}'.format(opt['transitions'],opt['AR_lags'],opt['K'],opt['downsample_factor'],timestr)
    fit_str = '{}-ARHMM_lag-{}_dsf-{}_kappa-{:.0e}_{}'.format(opt['transitions'],opt['AR_lags'],opt['downsample_factor'],opt['kappa'],timestr)
    
    #Create Save Directory
    SaveDirRoot = os.path.join(ResultsDir,fit_str)
    
    if not os.path.isdir(SaveDirRoot) and SAVE:
        os.makedirs(SaveDirRoot)

    return SaveDirRoot

## Make sub-directories for xval under the save directory root
def make_sub_dir(K, opt, kappa, i_fold, SAVE=True):   
    #e.g.: 'SaveDirRoot/K_05/'
    if i_fold == -1:
        SaveDir = os.path.join(opt['SaveDirRoot'], 'K-{:02d}'.format(K))
#         SaveDir = os.path.join(opt['SaveDirRoot'], 'Kappa-{:.0e}'.format(kappa))
    else:
        SaveDir = os.path.join(opt['SaveDirRoot'], 'K-{:02d}'.format(K),'kFold_{:02d}'.format(i_fold))
#         SaveDir = os.path.join(opt['SaveDirRoot'], 'Kappa-{:.0e}'.format(kappa),'kFold_{:02d}'.format(i_fold))
             
    if not os.path.isdir(SaveDir) and SAVE:
        os.makedirs(SaveDir)   

    print('Creating Directory: {}'.format(SaveDir))
    #e.g.: 'ARHMM-K_07-Robust~Sticky~InputHMM_101'
#     fname_sffx = '{}-ARHMM_lag-{}_K-{:02d}_Kappa-{:.0e}'.format(opt['transitions'],opt['AR_lags'],K,kappa)
    if opt['transitions'] == 'sticky':
        fname_sffx = 'sARHMM_lag-{}_K-{:02d}_dsf-{}_Kappa-{:.0e}'.format(opt['AR_lags'],K,opt['downsample_factor'],kappa)
    else:
        fname_sffx = 'ARHMM_lag-{}_K-{:02d}_dsf-{}'.format(opt['AR_lags'],K,opt['downsample_factor'])
    return SaveDir, fname_sffx

#Progress bar copied from https://github.com/mattjj/pyhsmm
def progprint(iterator,total=None,perline=25,show_times=True):
    times = []
    idx = 0
    if total is not None:
        numdigits = len('%d' % total)
    for thing in iterator:
        prev_time = time.time()
        yield thing
        times.append(time.time() - prev_time)
        sys.stdout.write('.')
        if (idx+1) % perline == 0:
            if show_times:
                avgtime = np.mean(times)
                if total is not None:
                    eta = sec2str(avgtime*(total-(idx+1)))
                    sys.stdout.write((
                        '  [ %%%dd/%%%dd, %%7.2fsec avg, ETA %%s ]\n'
                                % (numdigits,numdigits)) % (idx+1,total,avgtime,eta))
                else:
                    sys.stdout.write('  [ %d done, %7.2fsec avg ]\n' % (idx+1,avgtime))
            else:
                if total is not None:
                    sys.stdout.write(('  [ %%%dd/%%%dd ]\n' % (numdigits,numdigits) ) % (idx+1,total))
                else:
                    sys.stdout.write('  [ %d ]\n' % (idx+1))
        idx += 1
        sys.stdout.flush()
    print('')
    if show_times and len(times) > 0:
        total = sec2str(seconds=np.sum(times))
        print('%7.2fsec avg, %s total\n' % (np.mean(times),total))
        
def sec2str(seconds):
    hours, rem = divmod(seconds,3600)
    minutes, seconds = divmod(rem,60)
    if hours > 0:
        return '%02d:%02d:%02d' % (hours,minutes,round(seconds))
    elif minutes > 0:
        return '%02d:%02d' % (minutes,round(seconds))
    else:
        return '%0.2f' % seconds


##===== Utilty functions to interface with scott lindermans SSM code =====##

def get_hmm_latents(hmm, data_list):
    #Loop over trials and get the maximum a posteriori probability estimate of the latent states
    trMAPs = []; trPosteriors = []; trMasks = []

    for data in data_list:
        expected_states, expected_joints, log_likes = hmm.expected_states(data)

        trMAPs.append(np.argmax(expected_states,axis=1))
        trPosteriors.append(expected_states)
        trMasks.append(np.max(expected_states,axis=1) > 0.75)
    return trMAPs, trPosteriors, trMasks
    
def invert_perm(perm):
    inverse = np.zeros(len(perm),dtype=int)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def sort_states_by_usage(state_usage, trMAPs, trPosteriors):
    perm = np.argsort(-state_usage)
    invperm = invert_perm(perm)
    
    #Sort the MAP sequences and posteriors
    state_usage_sorted = state_usage[perm]
    trPosteriors_sorted = [Ez[:,perm] for Ez in trPosteriors]
    
    #Scott Linderman's code does give a state for the first timestep though
    trMAPs_sorted = [invperm[mapseq] for mapseq in trMAPs]
    
    return state_usage_sorted, trMAPs_sorted, trPosteriors_sorted, perm


def get_state_durations(trMAPs, trMasks, K, trial_indices=None):
    ##===== calculate state durations & usages in a given interval =====##  
    if trial_indices is None:
        nTrials = len(trMAPs)
        trial_indices = np.arange(0,nTrials)
    else:
        nTrials = len(trial_indices)
        
    # K+1 too, b/c measuring NaN state durations too.
    state_startend_list = [[[] for s in range(K+1)] for ii in range(nTrials)]
    state_duration_list = [[[] for s in range(K+1)] for ii in range(nTrials)]
    state_usage_pertrial = np.zeros((nTrials,K+1))
    state_usage_raw =  np.zeros(K+1)
    
    #Loop over trials
    for tr, iTrial in enumerate(trial_indices):
        #Apply threshold mask
        MAPcp = trMAPs[iTrial].copy()
        mask = trMasks[iTrial]
        MAPcp[~mask] = -1
#         pdb.set_trace()
        
        #Loop over states and nan-states
        for state in range(-1,K):
            state_1hot = np.concatenate(([0],(np.array(MAPcp) == state).astype(int),[0]))
            state_trans = np.diff(state_1hot)
            state_ends = np.nonzero(state_trans == -1)[0]
            state_starts = np.nonzero(state_trans == +1)[0]
            state_durations = state_ends - state_starts
            state_duration_list[tr][state] = state_durations
            state_usage_pertrial[tr,state] = np.sum(state_durations)
            state_startend_list[tr][state] = np.vstack((state_starts,state_ends)).T
            state_usage_raw[state] += np.sum(state_durations)
        
        #Normalize based on trial length 
#         state_usage_list[tr,:] = state_usage_list[tr,:]/np.sum(state_usage_list[tr,:])
            
    #Normalize State Usage
    state_usage = state_usage_raw/np.sum(state_usage_raw) 
    
    #Calculate mean state duration for each state
    mean_state_durations = [[] for s in range(K+1)]
    for state in range(-1,K):
        s_dur = []
        for iTrial in range(nTrials):
            s_dur.extend(state_duration_list[iTrial][state])
        mean_state_durations[state] = np.mean(s_dur)

    return (state_duration_list, state_startend_list, mean_state_durations, state_usage_pertrial, state_usage)

def get_transition_count_matrices(trMAPs, trMasks, K, normalize=True):
    ## Create 'Transition' count matrix for each trial from sparse MAP state sequence
    TCMs = np.zeros((len(trMAPs),K,K))
   
    #Loop over trials
    for iTrial, (map_seq,mask) in enumerate(zip(trMAPs,trMasks)):
        #Apply threshold mask
        MAP = np.squeeze(map_seq).copy()
        MAP[~mask] = -1 #instead of np.nan
        
        if all(~mask):
            continue
            
        #Lets drop all of the non states for easy sorting
        #Shorted MAP state sequence
        sMAP = [int(s) for s in MAP if s != -1]
        
        #Count 'state transitons' and fill matrix
        prev_state = sMAP[0]
        for iT,state in enumerate(sMAP):
            TCMs[iTrial,prev_state,state]+=1
            prev_state = state

        if normalize:
            TCMs[iTrial] = TCMs[iTrial]/np.sum(TCMs[iTrial])
        
    return TCMs

# Define randomized SVD function
def rSVD(X,r,q,p):
    # Step 1: Sample column space of X with P matrix
    ny = X.shape[1]
    P = np.random.randn(ny,r+p)
    Z = X @ P
    for k in range(q):
        Z = X @ (X.T @ Z)

    Q, R = np.linalg.qr(Z,mode='reduced')

    # Step 2: Compute SVD on projected Y = Q.T @ X
    Y = Q.T @ X
    UY, S, VT = np.linalg.svd(Y,full_matrices=0)
    U = Q @ UY
    
    # Calculate explained variancce
    X_trans = U * S
    exp_var = np.var(X_trans, axis=0)
    full_var = np.var(X,axis=0).sum()
    explained_variance_ratios = exp_var / full_var

    return U, S, VT, explained_variance_ratios