#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:53:52 2022

@author: phg17
"""

from os.path import join
import pandas as pd
import numpy as np
import collections
import mne
from .utils import get_list_events

path_paper = "/home/phg17/Documents/Paper3/"
path_data = "/home/phg17/Documents/Entrainment Experiment/Data"
path_epochs = "/home/phg17/Documents/Entrainment Experiment/Data Analysis/Epochs"
path_cluster = "/home/phg17/Documents/Entrainment Experiment/Data Analysis/Clusters"


def extract_behavioural(subject, parameter = 'Condition', selection = []):
    dict_tot = dict()
    result_file = join(path_data,subject + '_Entrainment', subject + '_Entrainment.csv')
    result_subject = pd.read_csv(result_file)
    for selection_value in selection:
        col, value = selection_value
        result_subject = result_subject[result_subject[col] == value]
    
    for trial in result_subject['Trial']:
        if parameter != 'Condition':
            if result_subject[parameter][trial] in dict_tot:
                dict_tot[str(result_subject[parameter][trial])].append(result_subject['Score'][trial])
            else:
                dict_tot[str(result_subject[parameter][trial])] = [result_subject['Score'][trial]]
        else:
            if result_subject['Type'][trial] == 'entrainment':
                if str(result_subject['Phase'][trial]) in dict_tot:
                    dict_tot[str(result_subject['Phase'][trial])].append(result_subject['Score'][trial])
                else:
                    dict_tot[str(result_subject['Phase'][trial])] = [result_subject['Score'][trial]]
            else:
                if result_subject['Type'][trial] in dict_tot:
                    dict_tot[str(result_subject['Type'][trial])].append(result_subject['Score'][trial])
                else:
                    dict_tot[str(result_subject['Type'][trial])] = [result_subject['Score'][trial]]
    
    return dict_tot


def compute_data(subject_list,  parameter = 'Condition', selection = []):
    data = []
    for subject in subject_list:
        result_subject = extract_behavioural(subject, parameter=parameter, selection=selection)
        ordered_result = collections.OrderedDict(sorted(result_subject.items()))
        data.append(np.mean(np.asarray(list(ordered_result.values())),axis=1))
    condition = list(ordered_result.keys())
    return condition, np.asarray(data)

def extract_tfr_diff(subjects_list, at_cond = 1 ,ref_cond = 'A',Fs = 100,
                     fmin = 1, fmax = 15, fstep = 0.25):
    
    delays = np.asarray([0,50,100,150])
    freqs = np.arange(fmin, fmax , fstep)
    n_cycles = freqs/2
    tfr_ref = []
    tfr_at = []
    tfr_diff = []
    
    
    for subject in subjects_list:
        epochs = mne.read_epochs(join(path_epochs,subject + '-epo.fif'))
        epochs.resample(Fs)
        epochs.crop(0,3)
        ref_events = get_list_events(conditions = [ref_cond])
        at_events = get_list_events(conditions = [str(at_cond)])
        ref_epochs = epochs[ref_events]
        at_epochs = epochs[at_events]
        
        _tfr_ref = mne.time_frequency.tfr_morlet(ref_epochs, freqs, n_cycles, 
                                                   return_itc=False, n_jobs=-1, 
                                                   output='power', average = True)
        _tfr_at = mne.time_frequency.tfr_morlet(at_epochs, freqs, n_cycles, 
                                                return_itc=False, n_jobs=-1,
                                                output='power', average = True)
        if at_cond == 'S':
            delay = 0
        else:
            delay = (delays*epochs.info['sfreq']/1000)[at_cond]
        tfr_ref.append(_tfr_ref.data)
        tfr_at.append(np.roll(_tfr_at.data, -int(delay),axis=2))
        
    tfr_ref = np.array(tfr_ref) # of shape [nsub, nch, nfreq, ntimes]
    tfr_at = np.array(tfr_at) # of shape [nsub, nch, nfreq, ntimes]
    tfr_diff = np.array(tfr_at) - np.array(tfr_ref) # of shape [nsub, nch, nfreq, ntimes]
    tfr_diff = np.array(tfr_diff)
    
    return freqs, tfr_ref, tfr_at, tfr_diff, _tfr_ref


