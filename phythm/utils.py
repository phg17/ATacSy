#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:53:07 2022

@author: phg17
"""

import pandas as pd
import numpy as np

def define_ROI(info,rois = [1,1,1,0,0], audio = False, tactile = False):
    '''
    In order: Left, Right, Centre, Back, Front
    '''

    # Calculate adjacency matrix between sensors from their locations
    #adjacency, ch_list = find_ch_connectivity(info, "eeg")
    ch_list = info['ch_names']
    #adjacency = adjacency.toarray()
    
    left_ROI = ['FT9','FT7','FC5','C5','CP5','TP7','TP9','T7','FC3','C3','CP3']
    right_ROI = ['FT10','FT8','FC6','FC4','T8','C6','C4','TP10','TP8','CP6','CP4']
    centre_ROI = ['FC1','FCz','FC2','C1','Cz','C2','CP1','CPz','CP2']
    back_ROI = ['O1','O2','Oz','P1','P2','P3','P4','P5','P6','P7','P8','PO3','PO4','PO7','PO8','POz','Pz']
    front_ROI = ['AF3','AF4','AF7','AF8','F1','F2','F3','F4','F5','F6','F7','F8','Fp1','Fp2','AFz']
    audio_ROI = ['FC5','FC3','C5','C3','FC6','FC4','C4','C6']
    tactile_ROI = ['FC1','FC3','F3','F1','CP5','CP3','P5','P3']
    ROI_electrodes = [audio_ROI,tactile_ROI]
    if audio and tactile:
        rois = [1,1]
    elif tactile:
        rois = [0,1]
    elif audio:
        rois = [1,0]
    else:
        ROI_electrodes = [left_ROI,right_ROI,centre_ROI,back_ROI,front_ROI]
    ROI_index = []
        
    for region, valid in zip(ROI_electrodes,rois):
        index_list = []
        for electrode in region:
            index_list.append(ch_list.index(electrode))
        ROI_index.append(index_list * valid)
        
    return ROI_index

def events_correspondance(events,result_subject,Fs = 1000):
    new_name_dict = define_events_namedict()
    events_info = np.asarray(events[0])
    new_events_info = []
    for event_index in range(1,events_info.shape[0]):
        single_event = events_info[event_index]
        single_info = result_subject.iloc[event_index-1]

        single_event[0] += int(single_info['Shift']/39062.5*Fs)
        single_event[2] = translate_event(single_info)
        new_events_info.append(single_event)
    new_events_info = np.asarray(new_events_info)
    return (new_events_info, new_name_dict)
        
def translate_event(trial):
    ID = 10000 + (trial['Gender'] == 'm') *1000 +\
    (trial['Type'] == 'sham') *100  +\
    (trial['Type'] == 'entrainment') * (int(trial['Phase']) == 0) * 200 +\
    (trial['Type'] == 'entrainment') * (int(trial['Phase']) == 1) * 300 +\
    (trial['Type'] == 'entrainment') * (int(trial['Phase']) == 3) * 400 +\
    (trial['Type'] == 'entrainment') * (int(trial['Phase']) == 4) * 500 +\
    (trial['Syllable'] == 'da') * 10 + (trial['Syllable'] == 'ga') * 20 +\
    (trial['Syllable'] == 'ka') * 30 + (trial['Syllable'] == 'pa') * 40 +\
    (trial['Syllable'] == 'ta') * 50 +\
    trial['Score']
    return ID

def define_events_namedict():
    new_name_dict = dict()
    for gender in ['f','m']:
        for condition in ['A','S','0','1','2','3']:
            for syllable in ['ba','da','ga','ka','pa','ta']:
                for score in ['0','1']:
                    trial = {'Gender':gender, 'Syllable' : syllable, 
                             'Score': int(score)}
                    if condition == 'A':
                        trial['Type'] = 'audio'
                        trial['Phase'] = 0
                    elif condition == 'S':
                        trial['Type'] = 'sham'
                        trial['Phase'] = 0
                    else:
                        trial['Type'] = 'entrainment'
                        trial['Phase'] = [0,1,3,4][int(condition)]
                    ID = translate_event(pd.Series(trial))
                    new_name_dict[gender + '_' + condition + '_' + syllable + '_' + score] = ID
    return new_name_dict

def get_list_events(genders = ['m','f'], conditions = ['A','S','0','1','2','3'], 
                    syllables = ['ba','da','ga','ka','pa','ta'], scores = ['0','1']):
    events_list = []
    for gender in genders:
        for condition in conditions:
            for syllable in syllables:
                for score in scores:
                    trial = {'Gender':gender, 'Syllable' : syllable, 
                             'Score': int(score)}
                    if condition == 'A':
                        trial['Type'] = 'audio'
                        trial['Phase'] = 0
                    elif condition == 'S':
                        trial['Type'] = 'sham'
                        trial['Phase'] = 0
                    else:
                        trial['Type'] = 'entrainment'
                        trial['Phase'] = [0,1,3,4][int(condition)]
                    ID = translate_event(pd.Series(trial))
                    events_list.append(str(ID))
    return events_list


def get_ROI():
    '''
    Based on https://www.researchgate.net/publication/338606840_Detecting_fatigue_in_car_drivers_and_aircraft_pilots_by_using_non-invasive_measures_The_value_of_differentiation_of_sleepiness_and_mental_fatigue/figures?lo=1
    '''
    ROI = {
            'F':['Fp1','Fp2','AF3','AFz','AF4','F1','Fz','F2'],
            'C':['FC1','FCz','FC2','C3','C1','Cz','C2','C4'],
            'P':['CP3','CP1','CPz','CP2','CP4','P5','P3','Pz','P2','P4','P6'],
            'O':['PO7','PO3','POz','PO4','PO8','O1','Oz','O2'],
            'LF':['AF7','F7','F5','F3','FT7','FC5','FC3','T7','C7','C5','TP7','CP5','P7'],
            'RF':['AF8','F4','F6','F8','FC4','FC6','FT8','C6','C8','T8','CP6','TP8','P8']}
    return ROI




            