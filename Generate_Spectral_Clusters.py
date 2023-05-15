import numpy as np
from os.path import join
from sklearn.preprocessing import scale
import phythm.utils as utils
import phythm.io as io
import phythm.stats as phystats
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle

path_epochs = "/home/phg17/Documents/Entrainment Experiment/Data Analysis/Epochs"
path_cluster = "/home/phg17/Documents/Entrainment Experiment/Data Analysis/Clusters"
subject_eeg = ['camembert','leader','policeman','frenchship','canada',
               'rotiqueen','favourite','kitten','2d','batman','pretzel2new',
               'laundry2','bearcub','stopwatch','tala']

for Fs in [100]:
    for tmin,tmax in [[1.2,1.3]]:
        for ref_cond in ['A']:
            for at_cond in [0,1,2,3,'S']:
                for frequency_band, region in [['theta','LF'],['theta','RF'],
                                               ['delta','LF'],['delta','RF'],
                                               ['alpha','P'],['alpha','F']]:
            
                    n_permutations = 20000
                    freqs, tfr_ref, tfr_at, tfr_diff, _tfr_ref = io.extract_tfr_diff(subjects_list = subject_eeg, 
                                                                      at_cond = at_cond, 
                                                                      ref_cond = ref_cond,
                                                                      Fs = Fs)
                    info = _tfr_ref.info
                    times = _tfr_ref.times
            
                    T_obs, clusters, p_values, H0 = phystats.cluster_frequency(tfr_ref, tfr_at, _tfr_ref, freqs, 
                                  info, times, 
                                  Fs = Fs, frequency_band = frequency_band,
                                  tmin = tmin, tmax = tmax,
                                  n_permutations = n_permutations, region = region)
                    print(np.min(p_values))
                    result = {
                            'tfr_ref' : np.mean(tfr_ref,axis=(0,2)),
                            'tfr_at' : np.mean(tfr_at,axis=(0,2)),
                            'info' : info,
                            'n_perm' : n_permutations,
                            'pvalues' : p_values}
                    
                    filename = 'cluster_' + str(at_cond) + '-' + str(ref_cond) + '_' + frequency_band + '_' + region + '_' + str(Fs) + '_' + str(tmin) + '-' + str(tmax) + '.pickle'
                    file = join(path_cluster, filename)
                    with open(file, 'wb') as handle:
                        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
