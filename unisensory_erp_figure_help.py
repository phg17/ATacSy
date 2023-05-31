import mne
import numpy as np
from os.path import join
from sklearn.preprocessing import scale
import phythm.utils as utils
import matplotlib.pyplot as plt
import scipy.stats as stats
from os.path import join

path_paper = '/home/phg17/Documents/Paper3'
path_epochs = "/home/phg17/Documents/Entrainment Experiment/Data Analysis/Epochs"
subject_eeg = ['camembert','leader','policeman','frenchship','canada',
               'rotiqueen','favourite','kitten','2d','batman','pretzel2new',
               'laundry2','bearcub','stopwatch','tala']

#%% Extract Events
at_cond = 1
delays = np.asarray([0,50,100,150])
audio_evokeds = []
at_evokeds = []
for subject in subject_eeg[:]:
    epochs = mne.read_epochs(join(path_epochs,subject + '-epo.fif'))
    epochs.resample(1000)
    epochs.filter(2,16)
    epochs.crop(-.5,2.5)
    audio_events = utils.get_list_events(conditions = ['A','S','0'])
    at_events = utils.get_list_events(conditions = [0,1,2,3])
    audio_epochs = epochs[audio_events]
    at_epochs = epochs[at_events]
    audio_evokeds.append(np.roll(np.mean(audio_epochs.get_data(),axis=0), shift = -int(1000*0.060), axis = 1))
    at_evokeds.append(np.roll(np.mean(at_epochs.get_data(),axis=0), shift = -int(1000*0.060), axis = 1))
info = epochs.info
delay = (delays*info['sfreq']/1000)[at_cond]
delay = 0
audio_epoched_array = np.asarray(audio_evokeds)
audio_epoch = mne.EpochsArray(audio_epoched_array,info, tmin=-0.5)
at_epoched_array = np.roll(np.asarray(at_evokeds), -int(delay),axis=2)
at_epoch = mne.EpochsArray(at_epoched_array,info, tmin=-0.5)

#%% Plot Tactile ERPs

tactile_erp = at_epoch.average()
tactile_erp.crop(-0.001,1.0)
fig = tactile_erp.plot_joint(times = [0.048,0.248,0.448,0.648,0.848])
axes = fig.get_axes()
axes[0].set_ylabel('Tactile ERPs (' +  u'\u03bc' + 'V)', fontsize = 22)
axes[0].set_xlabel('Time (ms)', fontsize = 22)
axes[1].set_xlabel('Peak 1', fontsize = 20)
axes[1].set_title(axes[1].title.get_text().split('.')[1][1:].replace('s','ms'), fontsize = 20)
axes[2].set_xlabel('Peak 2', fontsize = 22)
axes[2].set_title(axes[2].title.get_text().split('.')[1].replace('s','ms'), fontsize = 20)
axes[3].set_xlabel('Peak 3', fontsize = 22)
axes[3].set_title(axes[3].title.get_text().split('.')[1].replace('s','ms'), fontsize = 20)
axes[4].set_xlabel('Peak 4', fontsize = 22)
axes[4].set_title(axes[4].title.get_text().split('.')[1].replace('s','ms'), fontsize = 20)
axes[5].set_xlabel('Peak 5', fontsize = 22)
axes[5].set_title(axes[5].title.get_text().split('.')[1].replace('s','ms'), fontsize = 20)
axes[0].tick_params(axis='both', which='major', labelsize=20)
axes[0].tick_params(axis='both', which='minor', labelsize=20)
axes[0].ticklabel_format(axis='y',scilimits = (3,5))
axes[6].ticklabel_format(axis='y',scilimits = (3,5))
axes[6].tick_params(axis='both', which='major', labelsize=18)
axes[0].yaxis.get_offset_text().set_fontsize(20)
#axes[0].get_children()[68].set_text(' ' )
axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].set_xticklabels([0.1,0,200,400,600,800,1000])
fig.set_size_inches([15,8])
fig.tight_layout()
fig.savefig(join(path_paper, 'tactile_erp.svg'))

#%% Isolate Individual events

tactile_erp = at_epoch.average()
erp1 = tactile_erp.copy().detrend(1).crop(-0.,0.15).data
erp2 = tactile_erp.copy().detrend(1).crop(0.2,0.35).data
erp3 = tactile_erp.copy().detrend(1).crop(0.4,0.55).data
erp4 = tactile_erp.copy().detrend(1).crop(0.6,0.75).data
erp5 = tactile_erp.copy().detrend(1).crop(0.8,0.95).data
erp = np.mean(np.asarray([erp1,erp2,erp3,erp4,erp5]),axis=0)
evk = mne.EvokedArray(erp,info)
fig = evk.plot_joint(times = [0.0477])
axes = fig.get_axes()
axes[0].set_ylabel('Average Tactile ERP (' +  u'\u03bc' + 'V)', fontsize = 22)
axes[0].set_xlabel('Tactile lag (ms)', fontsize = 22)
axes[0].tick_params(axis='both', which='major', labelsize=20)
axes[0].tick_params(axis='both', which='minor', labelsize=20)
axes[1].set_xlabel('Peak 1', fontsize = 20)
axes[1].set_title(axes[1].title.get_text().split('.')[1][1:].replace('s','ms'), fontsize = 20)
axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[2].tick_params(axis='both', which='major', labelsize=18)
axes[0].set_xticklabels([0,20,40,60,80,100,120,140])

fig.tight_layout()
fig.set_size_inches([10,8])
fig.savefig(join(path_paper, 'tactile_erp_unique.svg'))

#%% Plot Audio ERPs

audio_erp = audio_epoch.average().detrend()
audio_erp.crop(0.95,1.25)
fig = audio_erp.plot_joint(times = [1.093,1.207])
axes = fig.get_axes()

axes[0].set_ylabel('Tactile ERPs (' +  u'\u03bc' + 'V)', fontsize = 22)
axes[0].set_xlabel('Syllable-lag (ms)', fontsize = 22)
axes[0].tick_params(axis='both', which='major', labelsize=20)
axes[0].tick_params(axis='both', which='minor', labelsize=20)
axes[1].set_xlabel('Peak 1', fontsize = 22)
axes[1].set_title(axes[1].title.get_text().split('.')[1][1:].replace('s','ms'), fontsize = 20)
axes[2].set_xlabel('Peak 2', fontsize = 22)
axes[2].set_title(axes[2].title.get_text().split('.')[1].replace('s','ms'), fontsize = 20)
axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[3].tick_params(axis='both', which='major', labelsize=18)
axes[0].set_xticklabels([-50,0,50,100,150,200,250])
fig.tight_layout()
fig.set_size_inches([9,8])
fig.savefig(join(path_paper, 'audio_erp.svg'))


