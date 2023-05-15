from os.path import join as join
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from mne.preprocessing.ica import ICA

import mne
import pickle

path_paper = "/home/phg17/Documents/Paper3/"
path_data = "/home/phg17/Documents/Entrainment Experiment/Data"
path_epochs = "/home/phg17/Documents/Entrainment Experiment/Data Analysis/Epochs"

size_title = 32
size_label = 18
size_ticks = 16
size_scatter = 25
size_box = 2.5
dpi_val = 2000
panel_size = [4.75,4.2857]
size_line = 3.5
boxwidths = 0.5

subject_name = ['camembert','leader','policeman','frenchship','canada','rotiqueen','favourite','kitten','2d',
                'batman','pretzel2new','laundry2','bearcub','stopwatch','tala'] #['peak','camembert','leader','policeman','frenchship','canada']

comparison_list = ['Syllable','Gender','Condition']
comparison = comparison_list[2]
Fs = 200

def gfp(evoked,averaged = True):
    if averaged:
        return np.sum((evoked.data) ** 2, axis=0)
        #return np.sum(evoked.data ** 2, axis=0)
    else:
        return np.sum(evoked.average().data ** 2, axis=0)
    
def func(x,A0,A1,sigma,phi):
    return A0 + 0.01*A1*np.cos((2*np.pi/sigma*x + phi))


#%% Epoching

cond_list = [0]

for subject in subject_name[:1]:
    result_file = join(path_data,subject + '_Entrainment',subject + '_Entrainment.csv')
    parameter_file = join(path_data,subject + '_Entrainment','parameters_1.npy')
    eeg_file = join(path_data,subject + '_Entrainment',subject + '.vhdr')
    eeg_preload = join(path_data,subject + '_Entrainment',subject + '_preload')
    result_subject = pd.read_csv(result_file)
    parameter_subject = np.load(parameter_file)
    raw = mne.io.read_raw_brainvision(eeg_file, preload = eeg_preload, verbose='ERROR')
    raw.resample(Fs, verbose='ERROR')
    raw.filter(l_freq=1,h_freq=32, verbose='ERROR') #0.5-12 1-12
    raw.set_eeg_reference('average', projection=True, verbose='ERROR')
    raw.apply_proj()
    
    
    n_components = 10
    ica = ICA(n_components = n_components, random_state = 97,max_iter=500)
    ica.fit(raw, verbose='ERROR')
    ica.detect_artifacts(raw)
    ica.apply(raw)

    info = raw.info
    channels_list = info['ch_names']
    info = raw.info
    for cond in cond_list:
        print(subject + ' ' +str(cond))
        util_dict = dict()
        util_dict['correct'] = []
        util_dict['syllable'] = []
        util_dict['condition'] = []
        
        events = mne.events_from_annotations(raw,'auto',verbose='ERROR')
        epochs = mne.Epochs(raw,events[0],tmin = -.1,tmax = 2.,preload=True, 
                            baseline=None,verbose='ERROR', detrend = 1)
        new_epochs = []
        button = []
        epochs.drop_bad()
        for i in range(1,495):
            if int(parameter_subject[i][0]) == cond:
                data = scale(epochs[i].get_data()[0],axis=1,with_std = False, with_mean = False)
                button.append(data[63,:])
                #data = scale(np.roll(data, -int(result_subject['Shift'][i]/39062.5*1000), axis=0),axis=1)
                data = np.roll(data, -int(result_subject['Shift'][i]/39062.5*info['sfreq']), axis=1)
                #print('shift: ' + str(result_subject['Shift'][i]/39062.5*1000))
                #button.append(data[63,:])
                new_epochs.append(data)
                util_dict['correct'].append(result_subject['Score'][i])
                util_dict['syllable'].append(result_subject['Syllable'][i])
                util_dict['condition'].append(int(parameter_subject[i][0]))
        epochs_cond = mne.EpochsArray(np.asarray(new_epochs),info,tmin=-0.01,
                                      verbose='ERROR')
        evoked_cond = epochs_cond.average()
        np.save(join(path_epochs,subject + '_evoked2.npy'),evoked_cond.data)
        np.save(join(path_epochs,subject + '_epochs2.npy'),epochs_cond.get_data())
        np.save(join(path_epochs,subject + '_' + str(cond) + '_epochs2.npy'),epochs_cond.get_data())
        np.save(join(path_epochs,subject + '_' + str(cond) + '_evoked2.npy'),evoked_cond.data)
        file = join(path_epochs,subject + '_' + str(cond) + '_utils2.pickle')
        output = open(file,'wb')
        pickle.dump(util_dict,output)
        output.close()
        file = join(path_epochs,subject + '_' + str(cond)  + '_utils2.pickle')
        output = open(file,'wb')
        pickle.dump(util_dict,output)
        output.close()
