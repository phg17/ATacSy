import phythm.io as io
import phythm.utils as utils
import phythm.viz as viz
import numpy as np
import matplotlib.pyplot as plt
import mne
from os.path import join
from phythm.viz import scabox
from phythm.stats import stat_effect, residual_cos, cosine, damped_cosine, residual_damped_cos
import pingouin as pg
import pandas as pd
import seaborn as sns
import matplotlib as mpl

path_paper = '/home/phg17/Documents/Paper3'
path_epochs = "/home/phg17/Documents/Entrainment Experiment/Data Analysis/Epochs"
subject_eeg = ['camembert','leader','policeman','frenchship','canada',
               'rotiqueen','favourite','kitten','2d','batman','pretzel2new',
               'laundry2','bearcub','stopwatch','tala']


#%% Load result
result = dict()
result_diff = dict()

for Fs in [100]:
    for ref_cond in ['A']:
        for at_cond in [0,1,2,3,'S']:
            result[at_cond] = dict()
            freqs, tfr_ref, tfr_at, tfr_diff, _tfr_ref = io.extract_tfr_diff(subjects_list = subject_eeg, 
                                                              at_cond = at_cond, 
                                                              ref_cond = ref_cond,
                                                              Fs = Fs)
            info = _tfr_ref.copy().info
            times = _tfr_ref.times
            freqs = _tfr_ref.freqs
            nave = _tfr_ref.nave
            
            result[at_cond] = tfr_at
            result_diff[at_cond] = tfr_diff
            result[ref_cond] = tfr_ref
                    
#%% Plot Power over Time

from sklearn.preprocessing import scale
color_dict = {0:'b', 1:'g', 2:'y',3:'r','S':'grey'}
cond_names = {0: '150 ms', 1: '200 ms', 2: '250 ms', 3: '300 ms', 'S': 'Irregular'}

ROI = utils.get_ROI()
region = 'RF'
ch_n = []
index_f_min = 20 #6 Hz
index_f_max = 21 #6.25 Hz
time_interest = 15

for name_chan in ROI[region]:
    try:
        ch_n.append(info['ch_names'].index(name_chan))
    except:
        print('no channel ' + name_chan)

avg_time = []
all_time= []
fig, ax = plt.subplots()       
#for cond in [1]:
for cond in [0,1,2,3,'S']:
    tfr_diff = result_diff[cond] 
    tfr_diff = np.mean(tfr_diff[:,ch_n],axis=1)[:,index_f_min:index_f_max,110:160]
    tfr_mean = np.mean(tfr_diff,axis=(0,1))
    tfr_std = np.std(np.mean(tfr_diff,axis=(1)),axis=0) / np.sqrt(tfr_diff.shape[0])
    time_vect = np.arange(tfr_mean.shape[0]) /100 + .06
    ax.plot(time_vect, tfr_mean, label = cond_names[cond], color = color_dict[cond])
    ax.fill_between(time_vect,tfr_mean - tfr_std, tfr_mean + tfr_std, 
                    alpha = 0.1, color = color_dict[cond])
    avg_time.append(np.mean(tfr_diff[:,:,time_interest],axis=1))
    all_time.append(tfr_mean)
    ax.scatter(time_interest / 100 + .06,np.mean(avg_time[-1]),
               color = color_dict[cond],alpha = .5,
               zorder = 5)
ax.legend(prop={'size': 14})

ax.set_xticks([0,.1,.2,.3,.4,.5])
ax.set_xticklabels([0,100,200,300,400,500],size = 20)
ax.set_yticklabels([-4,-3,-2,-1,0,1,2,3], size = 20)
ax.set_xlabel('Post-syllable lag (ms)', size=22)
ax.set_ylabel('GFP in band of interest (a.u.)', size=22)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xlim([.099,.55])
fig.tight_layout()
fig.set_size_inches([6.15,4.3])
fig.savefig(join(path_paper, 'power_time.svg'))

avg_time = scale(np.asarray(avg_time[:]), with_mean = False, with_std = False)
labels = ['-90','0','90','180']



#%% damped sine fitting

from scipy.optimize import least_squares
from phythm.stats import stat_effect, residual_cos, cosine
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm

x = np.arange(1000)/1000*400
cond = labels
data = scale(avg_time.T[:,:-1], axis = 1)

xData = np.repeat([150,200,250,300],data.shape[0])
yData = np.hstack(data.T)

x0 = np.asarray([0,100,166,0,124.5])
bounds_width = np.asarray([1e-10, 150, 0.1, 1.5, 1000])
bounds = (x0 - bounds_width,x0 + bounds_width)
res_lsq = least_squares(residual_damped_cos, x0, args=(xData, yData),
                        bounds = bounds, method='trf')

A0,A1,sigma,phi,tau = res_lsq.x

X1 = damped_cosine(xData,0,1,sigma,phi,tau)
#X = add_constant(X1)
X = X1

Y = yData #- A0

mod = sm.OLS(Y,X)
res2 = mod.fit()

mod = sm.RLM(Y, X, M=sm.robust.norms.HuberT())
res = mod.fit()



cond = [150,200,250,300]
fig3,ax3 = scabox(data,cond, positions = [150,200,250,300],
                  yticks = [-0.15,-0.1,-0.05,0,0.05,0.1,0.15],
                  boxwidths = 15, size_box = 2.5,
                  size_scatter = 25, size_fig = [7,5],
                  size_ticks = 16, size_label = 18)

if res.params.shape[0] == 1 and res.pvalues[0] < 0.05:
    ax3.plot(x,damped_cosine(x,A0,res.params[0],sigma,phi,tau), color = 'r', linewidth = 2, zorder = 5)
    ax3.plot(x,damped_cosine(x,A0,res.params[0],1e10,0,tau), '--r', linewidth = 1, zorder = 5)
    ax3.plot(x,-damped_cosine(x,A0,res.params[0],1e10,0,tau), '--r', linewidth = 1, zorder = 5)
elif res.params.shape[0] == 2 and res.pvalues[1] < 0.05:
    ax3.plot(x,damped_cosine(x,res.params[0],res.params[1],sigma,phi,tau), color = 'r', linewidth = 2, zorder = 5)
    ax3.plot(x,damped_cosine(x,res.params[0],res.params[1],1e10,0,tau), '--r', linewidth = 1, zorder = 5)
    ax3.plot(x,-damped_cosine(x,A0,res.params[0],1e10,0,tau), '--r', linewidth = 1, zorder = 5)
else:
    ax3.plot([x.min(),x.max()],[0,0],'r',linewidth = 2)
    ax3.plot(x,damped_cosine(x,A0,res.params[0],sigma,phi,tau), color = 'r', linewidth = 2, zorder = 5)
    ax3.plot(x,damped_cosine(x,A0,res.params[0],1e10,0,tau), '--r', linewidth = 2, zorder = 5)


ax3.plot([0,0],[-3,3],'--k' ,linewidth = 3)
ax3.set_xticks([0,50,100,150,200,250,300,350,400])
ax3.set_xticklabels([0,50,100,150,200,250,300,350,400], size = 18)
ax3.set_yticks([-3,-2,-1,0,1,2,3])
ax3.set_yticklabels([-3,-2,-1,0,1,2,3], size = 18)
#ax3.set_ylim([-2,2])
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.set_xlabel('Audio-tactile lag (ms)', size = 22)
ax3.set_ylabel('Neural audiotactile gain \n (z-score)', size = 22)

#ax3.scatter(xData, yData, color = 'grey', zorder = 5,s=size_scatter*2)
#ax3.scatter(xData, yData, color = 'k', zorder = 4,s=size_scatter*2.2, edgecolor = 'grey')

print(res2.summary())
print(res.summary())
print(A0,A1,sigma, phi, tau)
fig3.set_size_inches([6.15,4.3])
fig3.tight_layout()
fig3.savefig(join(path_paper,'new_fit_eeg.svg'))




#%% Correlation

cond, data = io.compute_data(subject_list=subject_eeg, parameter='Condition')
X1 = scale(np.hstack(scale(data[:,:4],axis=1)))
#X = add_constant(X1)
X = X1
Y = scale(np.hstack(avg_time.T[:,:4]))

Y_avg = (np.mean(avg_time,axis=1))
X_avg = (np.mean(scale(data[:,:4],axis=1),axis=0))

mod = sm.OLS(Y,X)
res = mod.fit()

rlm_model = sm.RLM(Y, X, M=sm.robust.norms.HuberT())
rlm_results = rlm_model.fit()


fig, ax = plt.subplots()
print(res.summary())
print(rlm_results.summary())
ax.scatter(X1,Y, color = 'k')
X_s = X1.reshape(15,4)
Y_s = Y.reshape(15,4)
for xavg,yavg,c in zip(X_s.T,Y_s.T,['k','k','k','k']):
    ax.scatter(xavg,yavg,s=150,color = c)
    
for xavg,yavg,c, label in zip(X_avg,Y_avg,
                              ['b','g','y','r'],
                              ['150 ms','200 ms', '250 ms', '300 ms']):
    ax.scatter(xavg,yavg,s=250,color = c, edgecolor = 'k', label = label)
ax.plot([-2,2],[0 + rlm_results.params[0] * -2.5, 
                     0 + rlm_results.params[0] * 2.5],
         color = 'red')
ax.set_xlabel('Syllable discrimination score \n (z-score)', size = 22)
ax.set_ylabel('Neural audiotactile gain \n (z-score)', size = 22)

ax.set_xticks([-2,-1,0,1,2])
ax.set_xticklabels([-2,-1,0,1,2], size = 20)
ax.set_yticks([-2,-1,0,1,2])
ax.set_yticklabels([-2,-1,0,1,2], size = 20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.legend(prop={'size': 16}, framealpha = 1)
fig.tight_layout()
fig.set_size_inches([10,10])
fig.savefig(join(path_paper,'Correlation.svg'))

#%% Repeated Measures Correlation
cond, data = io.compute_data(subject_list=subject_eeg, parameter='Condition')
X = scale(data[:,[0,1,2,3,5]],axis=1)
Y = scale(avg_time.T[:,:5],axis=1)

X = data[:,[0,1,2,3,5]]
Y = avg_time.T[:,:5]

dict_result = dict()
index_dict = 0
for subject_id in range(X.shape[0]):
    for measure in range(X.shape[1]):
        dict_result[index_dict] = {
                'subject_id' : int(subject_id + 1),
                'Behav' : X[subject_id, measure],
                'EEG' : Y[subject_id, measure]}
        index_dict += 1
df = pd.DataFrame(dict_result).T
print(pg.rm_corr(data=df, x='EEG', y='Behav', subject='subject_id'))
pg.plot_rm_corr(data=df, x='EEG', y='Behav', subject='subject_id', legend= True)

#%% Repeated Measures Correlation : Personnal figure
cond, data = io.compute_data(subject_list=subject_eeg, parameter='Condition')

X = data[:,[0,1,2,3,5]]
Y = avg_time.T[:,:5]

dict_result = dict()
index_dict = 0
for subject_id in range(X.shape[0]):
    for measure in range(X.shape[1]):
        dict_result[index_dict] = {
                'subject_id' : int(subject_id + 1),
                'Behav' : X[subject_id, measure],
                'EEG' : Y[subject_id, measure]}
        index_dict += 1
df = pd.DataFrame(dict_result).T
result_corr = pg.rm_corr(data=df, x='EEG', y='Behav', subject='subject_id')

r = result_corr['r']['rm_corr'] * 2e9
X_pos = np.mean(X,axis=1)
Y_pos = np.mean(Y, axis=1)

colormap = mpl.cm.nipy_spectral

fig,ax = plt.subplots()
for i in range(X.shape[0]):    
    color_index = int(i * 255 / X.shape[0])
    ax.scatter(Y[i,:],X[i,:], color = colormap(color_index), 
               s = 50,
               alpha = 0.8)
    ax.plot([np.min(Y[i,:]) , Y_pos[i] ,np.max(Y[i,:])], 
             [X_pos[i] - r*(Y_pos[i] - np.min(Y[i,:])), X_pos[i], X_pos[i] - r*(Y_pos[i] - np.max(Y[i,:]))],
             color = colormap(color_index), 
             linewidth = 2,
             alpha = 0.8)


ax.set_ylabel('Syllable discrimination score (%)', size = 22)
ax.set_xlabel('Neural audiotactile gain (a.u.)', size = 22)

ax.set_xticks([-6e-11,-4e-11,-2e-11,0,2e-11,4e-11,6e-11,8e-11])
ax.set_xticklabels([-6,-4,-2,0,2,4,6,8], size = 20)
ax.set_yticks([0.40,0.50,0.6,0.7,0.8])
ax.set_yticklabels([40,50,60,70,80], size = 20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.set_xlim([-2,2])
#ax.set_ylim([-2,2])

fig.tight_layout()

fig.set_size_inches([10,10])
fig.savefig(join(path_paper,'RM_correlation.svg'))

