import glob, json, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

import h5py
import sys
sys.path.insert(0,os.path.expanduser('~/Research/Jumping/jumping_behavior/'))
import util
import io_dict_to_hdf5 as ioh5
import ssm



parser = argparse.ArgumentParser(description='HMM for jumping data')
parser.add_argument('--SaveDirRoot',type=str, default='~/Research/Jumping/ARHMM/results')
parser.add_argument('--ModelID',type=str, default='sticky-ARHMM_lag-1_dsf-2_kappa-1e+04_2020-11-16_0150')
parser.add_argument('--DataPath',type=str, default='~/Research/Jumping/ARHMM/data/jumping_data_RIS_030921.h5')
parser.add_argument('--K',type=int, default=12, help='Number of States')
args = parser.parse_args()


def get_hmm_latents(hmm, data_list):
    #Loop over trials and get the maximum a posteriori probability estimate of the latent states
    trMAPs = []; trPosteriors = []; trMasks = []

    for data in data_list:
        expected_states, expected_joints, log_likes = hmm.expected_states(data)

        trMAPs.append(np.argmax(expected_states,axis=1))
        trPosteriors.append(expected_states)
        trMasks.append(np.max(expected_states,axis=1) > 0.75)
    return trMAPs, trPosteriors, trMasks




def get_state_sequence(hmm, data_test, opt, inputs=None):
    """
    Compute the local MAP state (arg-max of marginal state probabilities at each time step)
    and overall state usages.
    thresh: if marginal probability of MAP state is below threshold, replace with np.nan
    (or rather output a mask array with nan's in those time steps)

    Also output average state usages and the marginal state probabilities
    """
    T = 0; ll_heldout = 0
    state_usage = np.zeros(hmm.K)
    trMAPs = []
    trPosteriors = []
    trMasks = []
    
    #Loop over data to obtain MAP sequence for each trial
    for index, data in enumerate(data_test):
        #Get state probabilities and log-likelihood 
        if opt['transitions'] == 'inputdriven':
            inputdata = inputs[index]
            Ez, _, ll = hmm.expected_states(data,input=inputs)
        else:
            Ez, _, ll = hmm.expected_states(data)
        
        #Update number of data points, state usage, and llood of data 
        T += Ez.shape[0]
        state_usage += Ez.sum(axis=0)
        ll_heldout += ll
        
        #maximum a posteriori probability estimate of states
        map_seq = np.argmax(Ez,axis=1)
        max_prob = Ez[np.r_[0:Ez.shape[0]],map_seq]
        
        #Save sequences
        trMAPs.append(map_seq)
        trPosteriors.append(Ez)
        trMasks.append(max_prob > opt['MAP_threshold'])
    
    #Normalize
    state_usage /= T
    
    #Get parameters from ARHMM object
    param_dict = util.params_to_dict(hmm.params, opt)
    
    return trMAPs, trPosteriors, trMasks, state_usage, ll_heldout, param_dict



##===== ===== =====##
##===== Start =====##
if __name__ == "__main__":

    SaveDirRoot = os.path.expanduser(os.path.join(args.SaveDirRoot,args.ModelID))

    data_df = pd.read_hdf(os.path.expanduser(args.DataPath))
    K = args.K
    #Get path to h5 file with model parameters
    fpath = glob.glob(os.path.expanduser(os.path.join(args.SaveDirRoot,args.ModelID,'K-{:02d}'.format(K),'fit_parameters-*.h5')))
    if len(fpath) == 0:
        raise Exception('h5 file not found')
        

    #Load results from model fit
    results = ioh5.load(fpath[0])

    arhmm_params = results['arhmm_params']
    opt = results['hyperparams']

    nTrials = len(data_df)

    #DLC tracking confidence threshold at which to mask out data
    confidence_threshold = 0.8

    dsf = opt['downsample_factor']
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

        #Downsample if required
        tmp = np.vstack(xy_list)[:,2:-2].T; tmp2 = tmp if dsf == 1 else tmp[::dsf,:]; data_list.append(tmp2)
        tmp = np.vstack(ll_list)[:,2:-2].T; tmp2 = tmp if dsf == 1 else tmp[::dsf,:]; mask_list.append(tmp2)

    #Get number of time points and components per experiment
    nT, dObs = data_list[0].shape
    nComponents = dObs; opt['dObs'] = dObs
    


    #Create new ARHMM object
    arhmm = ssm.HMM(K, opt['dObs'],
                observations=opt['observations'], observation_kwargs={'lags':opt['AR_lags']},
                transitions=opt['transitions'], transition_kwargs={'kappa':opt['kappa']}) 
    #Set model parameters to previous model fit
    #Init_state_distn
    arhmm.init_state_distn.log_pi0 = arhmm_params['init_state_distn']['P0']
    #Transition
    arhmm.transitions.log_Ps = arhmm_params['transitions']['log_Ps']
    #Observations
    arhmm.observations.As = arhmm_params['observations']['As']
    arhmm.observations.bs = arhmm_params['observations']['bs']

    # trMAPs, trPosteriors, trMasks = get_hmm_latents(arhmm,data_list)
    trMAPs, trPosteriors, trMasks, state_usage, ll_heldout2, params_dict = get_state_sequence(arhmm, data_list, opt)
    ll_data_list = arhmm.log_likelihood(data_list)

    fname_sffx = 'ARHMM_lag_{}_K_{:02d}_dsf_{}'.format(opt['AR_lags'],K,opt['downsample_factor'])
    ioh5.save(os.path.join(SaveDirRoot, 'RIS_Data_{}.h5'.format(fname_sffx)), 
          {'trMAPs':trMAPs, 'trPosteriors':trPosteriors,'trMasks':trMasks, 
           'arhmm_params' : params_dict,'state_usage':state_usage, 
           'hyperparams' : opt,'ll_data_list':ll_data_list})
        