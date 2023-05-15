#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:29:21 2022

@author: phg17
"""

from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
import numpy as np
from .utils import get_ROI
import mne

def stat_effect(data, ref, labels, testype = 'non-parametric', focus = []):
    if len(focus) == 0:
        focus = np.arange(data.shape[1])
    if 'non-parametric':
        if data.shape[1] > 2:
            presence_effect = stats.friedmanchisquare(*data.T)[1]
            if presence_effect > 0.05:
                print('No Statistically Relevant Difference between Condition : p=' + str(presence_effect) + '\n\n')
                return presence_effect
            else:
                print('Statistically Relevant Difference between Condition : p=' + str(presence_effect) + '\n\n')
        discoveries = []
        comparison = []
        for condition_index in focus:
            if condition_index != ref:
                discoveries.append(stats.wilcoxon(data[:,ref], data[:,condition_index])[1])
                comparison.append([labels[condition_index],labels[ref]])
        corrected_discoveries = multipletests(discoveries, method='bonferroni')
        if np.sum(corrected_discoveries[0]) == 0:
            print('No Statistically Significant Difference Found after Correction \n')
        else:
            for pval, couple in zip(corrected_discoveries[1], comparison):
                if pval < 0.05:
                    print('Statistically Relevant Difference between ' + 
                          couple[0] + ' and ' + couple[1] + 
                          ' at p=' + str(pval)  + '\n')

        return discoveries, comparison
    
def cluster_frequency(tfr_ref, tfr_at, _tfr_ref, freqs, 
                      info, times,
                      Fs = 100, frequency_band = 'theta',
                      tmin = 1.1, tmax = 1.6,
                      n_permutations = 500, region = 'RF'):
    if frequency_band == 'theta':
        frequency_mask = np.logical_and(4 <= freqs, freqs < 8)
    elif frequency_band == 'delta':
        frequency_mask = np.logical_and(1 <= freqs, freqs < 4)
    elif frequency_band == 'alpha':
        frequency_mask = np.logical_and(8 <= freqs, freqs < 12)
    elif frequency_band == '5':
        frequency_mask = np.logical_and(4 <= freqs, freqs < 6)
    elif frequency_band == 'delta-theta':
        frequency_mask = np.logical_and(1 <= freqs, freqs < 8)
    else:
        frequency_mask = freqs
        
    time_list = list((np.asarray([tmin,tmax])*Fs).astype(int))
    temporal_mask = np.arange(time_list[0],time_list[1])
    
    ROIs = get_ROI()
    channels_selection = ROIs[region] 
    chs = info['ch_names']
    chn_mask = np.array([ch in channels_selection for ch in chs])
    
    audio_ = tfr_ref[:, chn_mask, :, :][:, :, frequency_mask, :][:, :, :, temporal_mask]
    at_ = tfr_at[:, chn_mask, :, :][:, :, frequency_mask, :][:, :, :, temporal_mask]
    
    # transpose because the cluster test requires channels to be last
    audio_ = audio_.transpose(0,2,3,1)
    at_ = at_.transpose(0,2,3,1)
    
    X = audio_ - at_ # shape [nsub, ntimes, nfreq, nch]
    
    n_times, n_freqs =  audio_.shape[1:3]
    info_frontocentral = _tfr_ref.copy().pick_channels(channels_selection).info
    adj_coo = mne.stats.combine_adjacency(n_times, # regular lattice adjacency for times
                                          np.zeros((n_freqs, n_freqs)), # no adjacency between freq. bins
                                          mne.channels.find_ch_adjacency(info_frontocentral, 'eeg')[0]) # source adj.
    
    
    # threshold = stats.f.ppf(1 - 0.05, dfn=1, dfd=len(all_subj)-2)  # F distribution
    tfce = dict(start=0, step=.25)  # ideally start 0 and small step
    
    T_obs, clusters, p_values, H0 = mne.stats.permutation_cluster_1samp_test(X, threshold=tfce,
                                                                             n_permutations=n_permutations,
                                                                             adjacency=adj_coo,
                                                                             n_jobs=-1, out_type='mask')
    
    return T_obs, clusters, p_values, H0


def cosine(x,A0,A1,sigma,phi):
    return A0 + 0.01*A1*np.cos((2*np.pi/sigma*x + phi))

def damped_cosine(x,A0,A1,sigma,phi,tau):
    return A0 + 0.01*A1*np.cos((2*np.pi/sigma*x + phi))*np.exp(-x/tau)


def residual_cos(x, t, y):
    return cosine(t,x[0],x[1],x[2],x[3]) - y

def residual_damped_cos(x, t, y):
    return damped_cosine(t,x[0],x[1],x[2],x[3],x[4]) - y
