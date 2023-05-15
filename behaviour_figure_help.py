import os

path_run = '/home/phg17/Documents/Entrainment Experiment/Data Analysis'
os.chdir(path_run)

import phythm.io as io
from phythm.viz import scabox, plot_significance
from phythm.stats import stat_effect
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from sklearn.preprocessing import scale
from scipy.optimize import least_squares
from phythm.stats import stat_effect, residual_cos, cosine
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm

path_paper = '/home/phg17/Documents/Paper3'
behav_list = ['camembert','leader','bristol','policeman','frenchship',
                'preacher','pretzel2new','canada','rotiqueen',
                'kitten','batman','2d','laundry2','bearcub','stopwatch','tala'] 




#%% Average Scores of Subjects
cond, data = io.compute_data(subject_list=behav_list, parameter='Condition')
data_audio = data[:,4]
data_audio = data[:,1]

data_audio = data_audio * 100
names = (np.arange(data_audio.shape[0]) + 1).astype(str)

fig1,ax1 = plt.subplots()
ax1.bar(np.arange(data_audio.shape[0]),data_audio,color = 'k')
ax1.plot([-0.5, data_audio.shape[0]],[50,50],'--r', linewidth = 2, label = 'chance level')
ax1.set_xticks(np.arange(data_audio.shape[0]))
ax1.legend()
ax1.set_ylabel('Syllable Recognition (%)')
ax1.set_xlabel('Subject ID')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
fig1.tight_layout()
fig1.set_size_inches([7,4])
fig1.savefig(join(path_paper, 'figbehav1.svg'))

#%% Condition Comparison and Statistical Test
cond, data = io.compute_data(subject_list=behav_list, parameter='Condition')
data = data[:,[4,0,1,2,3,5]]
cond = ['Audio', '150 ms','200 ms','250 ms','300 ms' ,'Irregular']
fig2,ax2 = scabox(data,cond)

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_xlim(0.4,6.6)
plot_significance(ax2, [1,3], [0.815,0.825], '***')
plot_significance(ax2, [0.5,6.5], [0.85,0.87], '***')
fig2.tight_layout()
fig2.set_size_inches([7,4])
fig2.savefig(join(path_paper, 'figbehav2.svg'))

#%% Distrib

cond, data = io.compute_data(subject_list=behav_list, parameter='Condition')
data = data[:,[4,0,1,2,3,5]]
cond = ['Audio','150 ms','200 ms','250 ms','300 ms' ,'Irregular']
rank = list(np.argmax(data[:,:],axis=1))
count = [rank.count(1),rank.count(2),rank.count(3),rank.count(4),rank.count(5)]
fig3, ax3 = plt.subplots()
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.bar(np.arange(5),count, color = 'k')
ax3.set_xticklabels(cond, size = 16)
ax3.set_yticklabels([-2,0,2,4,6,8,10], size = 16)
ax3.set_ylabel('Occurences',size = 18)
fig3.tight_layout()
fig3.set_size_inches([7,4])
fig3.savefig(join(path_paper, 'distrib.svg'))
