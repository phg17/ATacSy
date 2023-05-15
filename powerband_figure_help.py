import numpy as np
import scipy.stats as stats
import mne
import pickle as pkl
import os
from os.path import join
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import phythm.utils as utils
import phythm.viz as viz
from os.path import join


path_cluster = "/home/phg17/Documents/Entrainment Experiment/Data Analysis/Clusters"
path_paper = '/home/phg17/Documents/Paper3'
path_epochs = "/home/phg17/Documents/Entrainment Experiment/Data Analysis/Epochs"
epochs = mne.read_epochs(join(path_epochs,'camembert' + '-epo.fif'))
info = epochs.info
#%% Topograph representation

fig = mne.viz.plot_sensors(info)
ax = fig.get_axes()[0]
fig.savefig(join(path_paper,'topo.svg'))

#%% Cluster representation

file_list = sorted(os.listdir(path_cluster))
pval_dict = dict()
evk_dict = dict()
all_choices = []
all_files = []
for file in file_list:
    filename = join(path_cluster, file)
    with open(filename, "rb") as input_file:
        e = pkl.load(input_file)
        condition = file.split('_')[1]
        band = file.split('_')[2]
        position = file.split('_')[3]
        Fs = file.split('_')[4]
        time_range = file.split('_')[5].replace('.pickle','')
        minp = np.min(e['pvalues'])
        n = len(e['pvalues'][e['pvalues']<=0.05])
        
        if Fs == '100' and condition[-1] == 'A' and time_range == '1.1-1.6' and (band == 'alpha' or band == 'theta' or band == 'delta'):
            all_choices.append(minp)
            all_files.append(condition +' ' + band + ' ' + position + ' ' + Fs + ' '  + time_range + ' ' + str(n) +' ' + str(minp))
            print(condition, band, position, Fs,  time_range, n, minp)
            
        if Fs == '100' and band == 'theta' and position == 'LF' and time_range == '1.1-1.6' and condition[-1] == 'A':
            pval_dict[condition[0]] = e['pvalues']
            evk_dict[condition[0]] = e['tfr_at']
            final_values = e['pvalues']
            at_erp = e['tfr_at']
            ref_erp = e['tfr_ref']
            info = e['info']
            n_times = round((np.asarray(time_range.split('-')).astype(float) * 100)[1] - (np.asarray(time_range.split('-')).astype(float) * 100)[0])
corrected_pval = multipletests(all_choices, method = 'bonferroni')[1]
print('\n', len(all_choices) ,'\n')

#%% Corrected pvalues
        
for i,j in zip(all_files,corrected_pval):
    if j < 0.05:
        print(i,j)
        

#%% Representation of cluster 2
freqs = np.arange(4, 9, 1)
n_freq = int(final_values.shape[0]/n_times/12)
cond_names = {'0': '150ms', '1': '200ms', '2': '250ms', '3': '300ms', 'S': 'Arythmic'}
for cond in ['1']:
    final_values = pval_dict[cond]
    reshaped_pval = final_values.reshape(n_freq,n_times,12)
    
    fig, ax, im = viz.plot_cluster(reshaped_pval, pval_lim_color = 0.03,
                                   vmin=0, vmax = 3)
    #fig, ax, im = viz.plot_cluster_rawpval(reshaped_pval, pval_lim_color = 0.03,
    #                               vmin=0.0, vmax = 0.05)
    times = (np.arange(5)/ int(Fs) * 10000 + 100).astype(int)
    ax.set_yticks([-0.5,3.5,7.5,11.5,15.5])
    ax.set_yticklabels(freqs, size = 20)
    ax.set_ylabel(cond_names[cond] + '\n Frequency(Hz)', fontsize = 22)
    
    

    ax.set_xticks(np.arange(reshaped_pval.shape[1]//10)*10 + 4)
    ax.set_xticklabels(times, size = 20)
    ax.set_xlabel('Post-syllable lag (ms)', size = 22)


    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([-np.log10(0.05),-np.log10(0.01),-np.log10(0.001)])
    cbar.set_ticklabels([0.05,0.01,0.001])
    cbar.set_label('p-values', size = 20)
    

    fig.tight_layout()
    im.colorbar.ax.tick_params(labelsize = 16)
    ax.set_xlim([3.99,45])
    fig.set_size_inches([7.1,4.3])
    
    fig.savefig(join(path_paper, cond + '_Cluster.svg'))



    

