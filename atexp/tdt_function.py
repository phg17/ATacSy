# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:13:53 2019

@author: phg17
"""

import numpy as np
import matplotlib.pyplot as plt
#import sounddevice as sd
import scipy.io.wavfile as sio
import scipy.signal as sig
import time
import os
from scipy.signal.signaltools import hilbert, filtfilt
from scipy.signal.filter_design import butter
from time import sleep
import random
from tdt import DSPProject
from pandas_ods_reader import read_ods
from scipy.stats import norm
from scipy.sparse import load_npz

Conditions_behavioural = dict()
Conditions_behavioural[0] = {'type':'audio','delay':0,'correlated':True} #audio only
Conditions_behavioural[1] = {'type':'audio-tactile','delay':60,'correlated':True} #audio tactile -60 (late tactile)
Conditions_behavioural[2] = {'type':'audio-tactile','delay':0,'correlated':True} #audio tactile 0 (sync tactile)
Conditions_behavioural[3] = {'type':'audio-tactile','delay':-60,'correlated':True} #audio tactile 60 (tactile in advance)
Conditions_behavioural[4] = {'type':'audio-tactile','delay':-120,'correlated':True} #audio tactile 120 (tactile in advance)
Conditions_behavioural[5] = {'type':'audio-tactile','delay':-180,'correlated':True} #audio tactile 180 (tactile in advance)
Conditions_behavioural[6] = {'type':'audio-tactile','delay':0,'correlated':False} #uncorrelated (tactile uncorrelated)

Conditions_EEG = dict()
Conditions_EEG[0] = {'type':'audio','delay':0,'correlated':True} #audio only
Conditions_EEG[1] = {'type':'tactile','delay':0,'correlated':True} #tactile only
Conditions_EEG[2] = {'type':'audio-tactile','delay':60,'correlated':True} #audio tactile -60 (late tactile)
Conditions_EEG[3] = {'type':'audio-tactile','delay':0,'correlated':True} #audio tactile 0 (sync advance)
Conditions_EEG[4] = {'type':'audio-tactile','delay':-60,'correlated':True} #audio tactile 60 (tactile in advance)
Conditions_EEG[5] = {'type':'audio-tactile','delay':-120,'correlated':True} #audio tactile 120 (tactile in advance)
Conditions_EEG[6] = {'type':'audio-tactile','delay':-180,'correlated':True} #audio tactile 120 (tactile in advance)
Conditions_EEG[7] = {'type':'audio-tactile','delay':0,'correlated':False} #uncorrelated (tactile uncorrelated)

Conditions_Entrainment = dict()
Conditions_Entrainment[0] = {'type':'audio','frequency':0,'phase':0} #audio only 
Conditions_Entrainment[1] = {'type':'entrainment','frequency':4,'phase':0} #2 Hz at phase 2
Conditions_Entrainment[2] = {'type':'entrainment','frequency':4,'phase':np.pi/2} #2 Hz at phase 3
Conditions_Entrainment[3] = {'type':'entrainment','frequency':4,'phase':np.pi} #2 Hz at phase 4
Conditions_Entrainment[4] = {'type':'entrainment','frequency':4,'phase':3*np.pi/2} #audio only 
Conditions_Entrainment[5] = {'type':'sham','frequency':0,'phase':0} #random only with unif distrib of times
Syllables_pair = {'pa':'ba','ba':'pa','da':'ta','ta':'da','ga':'ka','ka':'ga'}



def Circuit_Setup(circuit_name):
    project = DSPProject()
    circuit = project.load_circuit(circuit_name, 'RX8')
    fs_circuit=circuit.fs
    return project, circuit, fs_circuit
    


def GenerateData(FILENAME):
    """ function to takes a wav file and returns the sample rate and the signal,
    returns a mono audio signal if two channels are provided."""
    Fs, Sig = sio.read(FILENAME)
    is_mono = len(Sig.shape) == 1 or Sig.shape[-1] == 1
    if is_mono:
        return Fs, Sig
    else:
        return Fs, Sig[:, 0]

def Single_Tactile(Fs,length=2000):
    wide = 0.0075*Fs
    y = np.zeros(length)
    x = np.arange(length)
    loc = int(length/2)
    scale = wide
    y += norm.pdf(x,loc,scale)
    carry_tone = np.sin(np.arange(wide*12)/Fs*80*2*np.pi)
    a=np.argmax(carry_tone[750:])
    a+=750
    b=np.argmax(carry_tone[a:])
    b+=a
    y *= carry_tone[b-loc:b+loc]
    y /= max(y)
    return y

def Pulse_Series(Fs,pause,n_pulses):
    length = n_pulses * (pause + 2000)
    pause_vect = np.zeros(pause)
    output = []
    for i in range(n_pulses):
        output += list(pause_vect)
        output += list(Single_Tactile(Fs))
    return output,length

def RMS(signal):
    return np.sqrt(np.mean(np.power(signal, 2)))

def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)
    
def AddNoise(Target, Noise, SNR, spacing):
    l_Noise = len(Noise)
    l_Target = len(Target)
    #spacing = 2000
    rmsS = RMS(Target)
    rmsN = rmsS*(10**(-SNR/20.))
    #insert = np.random.randint(0,l_Noise - l_Target - 2 * spacing)
    insert = 0
    Noise = Noise[insert:insert + 2 * spacing + l_Target]
    Noise = Noise * rmsN
    Target_Noise = Noise
    Target_Noise[spacing:spacing + l_Target] += Target
    Target_Noise = Target_Noise / RMS(Target_Noise)
    
    return Target_Noise

def AddNoisePostNorm(Target, Noise, SNR, spacing):
    l_Target = len(Target)
    rmsS = 1
    rmsN = rmsS*(10**(-SNR/20.))
    insert = 0
    Noise = Noise[insert:insert + 2 * spacing + l_Target]
    Noise = Noise * rmsN
    Target_Noise = Noise
    Target_Noise[spacing:spacing + l_Target] += Target
    Target_Noise = Target_Noise / RMS(Target_Noise)
    
    return Target_Noise



def hilbert_envelope(data, fs_circuit, phase_shift=0.):
    "Take filename return PHASE SHIFTED envelope + sound in correct format..."
    "Take filename return PHASE SHIFTED envelope + sound in correct format..."

    f_cutoff = 20. # in Hertz
    Wn = f_cutoff / (fs_circuit/2.)
    #Butterworth LP filter
    b, a = butter(4, Wn)

    # Envelope extraction:
    envelope = np.abs(hilbert(data/RMS(data)))

    filt_sig = filtfilt(b, a, envelope)
    Hilb_sig2 = hilbert(filt_sig.real)

    Hilb_sig2_phase_shift = Hilb_sig2 * np.exp(1j * phase_shift * np.pi/3)
    final = np.real(Hilb_sig2_phase_shift)
    
    return final


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y







class Stimuli_EEG():
    def __init__(self, chapter, part, condition,snr, circuit):
        
        start = time.time()
        
        #Genetal Info
        
        self.circuit = circuit
        self.type = Conditions_EEG[condition]['type']
        self.delay = Conditions_EEG[condition]['delay']
        self.correlated = Conditions_EEG[condition]['correlated']
        self.Fs = 39062.5
        self.snr = snr

        
        #Load Files
        path_to_noise = os.path.join(r'D:/Signal_Pierre/','ssn')
        path = 'D:\Signal_Pierre\stimuli\Odin'
        audiofile = 'Odin_' + str(chapter) + '_' + str(part) + '_audio.npy'
        tactilefile = 'Odin_' + str(chapter) + '_' + str(part) + '_phone_.npy'
        questionsfile = 'Odin_' + str(chapter) + '_Questions.ods'
        summaryfile = 'Odin_' + str(chapter) + '_' + str(part) + '_Summary.txt'
        
        targetAUDIO = os.path.join(path,audiofile)
        targetTACTILE = os.path.join(path,tactilefile)
        targetNOISE = os.path.join(path_to_noise,'ODIN_SSN.npy')
        #targetNOISE = os.path.join(path_to_noise,'ssn_1.npy')
        
        Audio = np.load(targetAUDIO)
        Tactile = np.load(targetTACTILE)
        Noise = np.load(targetNOISE)
        
        #Scale Signals

        Tactile = Tactile / max(Tactile) * 1.4
        Noise = Noise / RMS(Noise)
        #Noise = np.asarray(list(Noise) + list(Noise) + list(Noise) + list(Noise) + list(Noise) + list(Noise))[0:len(Audio)]
        Noise = np.asarray(list(Noise))[0:len(Audio)]
        self.noise = Noise
        Noise_Audio = AddNoisePostNorm(Audio,Noise,snr,spacing=0) 
        
        
        #Handle Questions and Summary
        targetQuestions = os.path.join(path,questionsfile)
        targetSummary = os.path.join(path,summaryfile)
        f = open(targetSummary)
        self.Summary = f.read()
        f.close()
        self.Questions = read_ods(targetQuestions,1,headers = False).iloc[:,part-1]
        self.Answers=[0,0,0,0]
        
        #Duration Info
        self.length = len(Audio)
        self.clean_audio = np.load(targetAUDIO) / 500
        self.duration = round(self.length/39062.5,2)
        
        #Handle different conditions
        if self.type == 'tactile': 
            self.audio = np.zeros(self.length)
        else:
            self.audio = Noise_Audio / 500
        
        if self.type == 'audio':
            self.tactile = np.zeros(self.length)
        elif not self.correlated:
            self.tactile = np.roll(Tactile, int(self.length/4))
        else:
            self.tactile = np.roll(Tactile, int(self.delay / 1000 * self.Fs))
        
        self.clean_tactile = Tactile
        
        
        self.timescale = np.arange(self.length) / self.Fs
        
        #Handle tactile task
        if self.type == 'tactile':
            Series, l_Series = Pulse_Series(self.Fs,14000,5)
            Series = np.asarray(Series) * 1.4
            self.task = []
            for i in range(10):
                self.task.append(np.random.randint(int(self.length/10*i + 10000),int(self.length/10*(i+1) - l_Series)))
            for i in self.task:
                self.tactile[i : i + l_Series] = Series
                #self.tactile[i - 1000 : i + 1000] = np.sin(2 * np.pi * 90 * np.arange(0,2000) /39062.5)
        else:
            self.task = None
        
        #Handle Information to save
        self.info = dict()
        self.info['Fs'] = self.Fs
        self.info['type'] = self.type
        self.info['delay'] = self.delay
        self.info['correlated'] = self.correlated
        self.info['file'] = str(chapter) + '_' + str(part)
        self.info['task'] = self.task
        
        print('Stimuli generated in ', str(round(time.time() - start,2)), ' seconds')
        
        
    def check(self):
        if (len(self.audio) == len(self.tactile) == self.length) :
            print('All Stimuli have the same length')
        else:
            raise Exception('All Stimuli do not have the same length')
        if (self.Fs == self.circuit.fs):
            print('Consistent sampling frequency: ', str(self.Fs))
        else:
            raise Exception('Inconsistent Sampling Frequency')
        
    def plot(self, *stim, new_window = True):
        start = time.time()
        if new_window:
            plt.figure()
        if len(stim) == 0:
            plt.subplot(211)
            plt.plot(self.timescale,self.audio)
            plt.subplot(212)
            plt.plot(self.timescale,self.tactile)
        else:
            for i in stim:
                if i == 'audio':
                    plt.plot(self.timescale,self.audio)
                elif i == 'tactile':
                    plt.plot(self.timescale,self.tactile/500)
                else:
                    raise Exception('Invalid Condition')
        
        
        print('Graph generated in ', str(round(time.time() - start,2)), ' seconds')
        
    def load_into_buffer(self):
        start = time.time()
        
        self.check()
        
        audio_in_buffer = self.circuit.get_buffer('audio_in', 'w')
        tactile_in_buffer = self.circuit.get_buffer('tactile_in', 'w')
        #stimtrack_in_buffer = self.circuit.get_buffer('stimtrack_in', 'w')
        #trigger_in_buffer = self.circuit.get_buffer('trigger_in', 'w')


        self.circuit.set_tag('size_audio', self.length)
        self.circuit.set_tag('size_tactile', self.length)
        #self.circuit.set_tag('size_stimtrack', self.length)
        #self.circuit.set_tag('size_trigger', self.length - 200)
        audio_in_buffer.write(self.audio)
        tactile_in_buffer.write(self.tactile)
        #stimtrack_in_buffer.write(self.stimtrack)
        #trigger_in_buffer.write(self.trigger)
    

        load=time.time()
        print('Stimuli loaded in: ',str(load-start),' seconds')
        
        
    def partial_load(self, duration = 15):
        short_length = int(duration * 39062.5)
        start = time.time()
        
        self.check()
        
        audio_in_buffer = self.circuit.get_buffer('audio_in', 'w')
        tactile_in_buffer = self.circuit.get_buffer('tactile_in', 'w')
        #stimtrack_in_buffer = self.circuit.get_buffer('stimtrack_in', 'w')
        #trigger_in_buffer = self.circuit.get_buffer('trigger_in', 'w')


        self.circuit.set_tag('size_audio', short_length)
        self.circuit.set_tag('size_tactile', short_length)
        #self.circuit.set_tag('size_stimtrack', self.length)
        #self.circuit.set_tag('size_trigger', short_length - 200)
        audio_in_buffer.write(self.audio[0:short_length])
        tactile_in_buffer.write(self.tactile[0:short_length])
        #stimtrack_in_buffer.write(self.stimtrack)
        #trigger_in_buffer.write(self.trigger[0:short_length])
    

        load=time.time()
        print('Stimuli loaded in: ',str(load-start),' seconds')
      
    def partial_send(self, duration = 15):
        short_length = int(duration * 39062.5)
        start = time.time()
        self.circuit.start()
        t0= time.time()
        while 1:
            #if time.time()-t0 > self.task[0]/39062.5 - 1 and jambon == False:
            #    print('JAMBOM')
            #    jambon = True
            if (time.time()-t0) + 0.3 > short_length / self.circuit.fs:
                break

        self.circuit.stop()
        
        end = time.time()
        print('Stimuli was sent for: ',str(end - start),' seconds')
        self.circuit.set_tag('audio_in_i', 0)
        self.circuit.set_tag('tactile_in_i', 0)
        #self.circuit.set_tag('stimtrack_in_i', 0)
        #self.circuit.set_tag('trigger_in_i', 0)        
        
    def send(self):
        start = time.time()
        self.circuit.start()
        t0= time.time()
        while 1:
            if (time.time()-t0) + 0.3 > self.length / self.circuit.fs:
                break

        self.circuit.stop()
        
        end = time.time()
        print('Stimuli was sent for: ',str(end - start),' seconds')
        self.circuit.set_tag('audio_in_i', 0)
        self.circuit.set_tag('tactile_in_i', 0)
        #self.circuit.set_tag('stimtrack_in_i', 0)
        #self.circuit.set_tag('trigger_in_i', 0)
        
        
        
        
        

class Stimuli_Behavioural():
    def __init__(self, sentence, condition, loopTABLE, snr, circuit):
        
        start = time.time()
        
        #Genetal Info
        
        self.circuit = circuit
        self.type = Conditions_behavioural[condition]['type']
        self.delay = Conditions_behavioural[condition]['delay']
        self.correlated = Conditions_behavioural[condition]['correlated']
        self.Fs = 39062.5
        self.snr = snr

        
        #Load Files
        path_to_noise = os.path.join(r'D:/Signal_Pierre/','ssn')
        path = 'D:\Signal_Pierre\stimuli\Stimuli_Behavioural'
        
        FILEid = '{:03}'.format(sentence)
        FILEid2 = '{:03}'.format(np.random.randint(1,2000))
        
        audiofile = 'sent_4k__%s_audio.npy' %FILEid
        tactilefile = 'sent_4k__%s_phone_.npy' %FILEid
        tactilefile_2 = 'sent_4k__%s_phone_.npy' %FILEid2
        self.real_sentence = loopTABLE['Real Sentence'][sentence-1]
        
        targetAUDIO = os.path.join(path,audiofile)
        targetTACTILE = os.path.join(path,tactilefile)
        targetTACTILE2 = os.path.join(path,tactilefile_2)
        targetNOISE = os.path.join(path_to_noise,'ssn_1.npy')
        
        Audio = np.load(targetAUDIO)
        Tactile = np.load(targetTACTILE)
        Tactile_2 = np.load(targetTACTILE2)
        Noise = np.load(targetNOISE)
        
        #Scale Signals
        Tactile = Tactile / max(Tactile) * 1.3
        Tactile_2 = Tactile_2 / max(Tactile_2) * 1.3
        Noise = Noise / RMS(Noise)
        Noise = np.asarray(list(Noise) + list(Noise) + list(Noise) + list(Noise) + list(Noise) + list(Noise))[0:len(Audio)]
        Noise_Audio = AddNoisePostNorm(Audio,Noise,snr,spacing=0) 
        
        
        #Duration Info
        self.length = len(Audio)
        self.clean_audio = np.load(targetAUDIO) / 500
        self.duration = round(self.length/39062.5,2)
        
        #Handle different conditions
        self.audio = Noise_Audio / 500

        
        if self.type == 'audio':
            self.tactile = np.zeros(self.length)
        elif not self.correlated:
            print(len(Tactile_2))
            print(len(Tactile))
            print(self.length)
            if len(Tactile_2) > self.length:
                Tactile_2 = Tactile_2[0:self.length]
            else:
                Tactile_2 = np.pad(Tactile_2,(0,len(Tactile) - len(Tactile_2)), 'constant', constant_values=(0,0))
            print(len(Tactile_2))
            print(len(Tactile))
            print(self.length)            
            self.tactile = Tactile_2
            
        else:
            self.tactile = np.roll(Tactile, int(self.delay / 1000 * self.Fs))
        
        
        self.timescale = np.arange(self.length) / self.Fs
        
        
        #Handle Information to save
        self.info = dict()
        self.info['Fs'] = self.Fs
        self.info['type'] = self.type
        self.info['delay'] = self.delay
        self.info['correlated'] = self.correlated
        self.info['file'] = sentence

        
        print('Stimuli generated in ', str(round(time.time() - start,2)), ' seconds')
        
        
    def check(self):
        if (len(self.audio) == len(self.tactile) == self.length) :
            print('All Stimuli have the same length')
        else:
            raise Exception('All Stimuli do not have the same length')
        if (self.Fs == self.circuit.fs):
            print('Consistent sampling frequency: ', str(self.Fs))
        else:
            raise Exception('Inconsistent Sampling Frequency')
        
    def plot(self, *stim, new_window = True):
        start = time.time()
        if new_window:
            plt.figure()
        if len(stim) == 0:
            plt.subplot(211)
            plt.plot(self.timescale,self.audio)
            plt.subplot(212)
            plt.plot(self.timescale,self.tactile)
        else:
            for i in stim:
                if i == 'audio':
                    plt.plot(self.timescale,self.audio)
                elif i == 'tactile':
                    plt.plot(self.timescale,self.tactile/500)
                else:
                    raise Exception('Invalid Condition')
        
        
        print('Graph generated in ', str(round(time.time() - start,2)), ' seconds')
        
    def load_into_buffer(self):
        start = time.time()
        
        self.check()
        
        audio_in_buffer = self.circuit.get_buffer('audio_in', 'w')
        tactile_in_buffer = self.circuit.get_buffer('tactile_in', 'w')



        self.circuit.set_tag('size_audio', self.length)
        self.circuit.set_tag('size_tactile', self.length)
        audio_in_buffer.write(self.audio)
        tactile_in_buffer.write(self.tactile)

    

        load=time.time()
        print('Stimuli loaded in: ',str(load-start),' seconds')
        
    def send(self):
        start = time.time()
        self.circuit.start()
        t0= time.time()
        while 1:
            if (time.time()-t0) + 0.3 > self.length / self.circuit.fs:
                break

        self.circuit.stop()
        
        end = time.time()
        print('Stimuli was sent for: ',str(end - start),' seconds')
        self.circuit.set_tag('audio_in_i', 0)
        self.circuit.set_tag('tactile_in_i', 0)

        
    def save_info(self,filesave,answers):
        
        #datafile = open(summary_file,'r')
        #summary = datafile.read()
        #datafile.close()
        print('jambon')

        

class Stimuli_Tactile():
    def __init__(self, circuit):
        
        start = time.time()
        
        #Genetal Info
        self.circuit = circuit
        self.Fs = 39062.5
        
        #Load Files
        path = 'D:\Signal_Pierre\stimuli\Other'
        file = '5_peaks.npy'
        targetTACTILE = os.path.join(path,file)
        Tactile = np.load(targetTACTILE)
        
        #Scale Signals

        Tactile = Tactile / max(Tactile) * 1.4
        self.length = len(Tactile)
        self.tactile = Tactile
        
        self.clean_tactile = Tactile
        
        self.timescale = np.arange(self.length) / self.Fs
        print('Stimuli generated in ', str(round(time.time() - start,2)), ' seconds')

        
    def load_into_buffer(self):
        start = time.time()

        tactile_in_buffer = self.circuit.get_buffer('tactile_in', 'w')
        self.circuit.set_tag('size_tactile', self.length)
        tactile_in_buffer.write(self.tactile)
        load=time.time()
        print('Stimuli loaded in: ',str(load-start),' seconds')
        
        
        
    def send(self):
        start = time.time()
        self.circuit.start()
        t0= time.time()
        while 1:
            if (time.time()-t0) + 0.3 > self.length / self.circuit.fs:
                break

        self.circuit.stop()
        
        end = time.time()
        print('Stimuli was sent for: ',str(end - start),' seconds')
        self.circuit.set_tag('tactile_in_i', 0)


class Stimuli_Bench():
    def __init__(self, duration,snr, circuit):
        
        start = time.time()
        chapter = 2
        part = 1
        n_samples = int(duration * 39062.5)
        #Genetal Info
        
        self.circuit = circuit
        self.Fs = 39062.5
        self.snr = snr

        
        #Load Files
        path_to_noise = os.path.join(r'D:/Signal_Pierre/','ssn')
        path = 'D:\Signal_Pierre\stimuli\Odin'
        audiofile = 'Odin_' + str(chapter) + '_' + str(part) + '_audio.npy'
        tactilefile = 'Odin_' + str(chapter) + '_' + str(part) + '_phone_.npy'
        phonemefile = 'Odin_' + str(chapter) + '_' + str(part) + '_compressed_phonemes.npz'
        phoneticfile = 'Odin_' + str(chapter) + '_' + str(part) + '_compressed_phonetic_features.npz'
        
        
        targetAUDIO = os.path.join(path,audiofile)
        targetTACTILE = os.path.join(path,tactilefile)
        targetNOISE = os.path.join(path_to_noise,'ODIN_SSN.npy')
        targetPHONEMES =  os.path.join(path,phonemefile)
        targetPHONETIC =  os.path.join(path,phoneticfile)
        #targetNOISE = os.path.join(path_to_noise,'ssn_1.npy')
        
        Audio = np.load(targetAUDIO)[:n_samples]
        Noise = np.load(targetNOISE)[:n_samples]
        phonemes = load_npz(targetPHONEMES)[:n_samples]
        phonetic = load_npz(targetPHONETIC)[:n_samples]
        Noise = Noise / RMS(Noise)
        Noise = np.asarray(list(Noise) + list(Noise) + list(Noise) + list(Noise) + list(Noise) + list(Noise))[0:len(Audio)]
        Noise_Audio = AddNoisePostNorm(Audio,Noise,snr,spacing=0) 
        self.audio = Noise_Audio / 500
        self.phonemes = phonemes.toarray()[:,:n_samples]
        self.phonetic = phonetic.toarray()[:,:n_samples]
        self.Front = self.phonetic[13,:]
        self.Central = self.phonetic[14,:]
        self.Back = self.phonetic[15,:]
        

        timing_back  = []
        timing_front = []
        for i in range(n_samples):
            if self.Back[i] == 1:
                timing_back.append(i)
            if self.Front[i] == 1:
                timing_front.append(i)
        wide_back = int(0.01 * 39062.5)
        wide_front = int(0.0050 * 39062.5)
        x = np.arange(len(Audio))
        y = np.zeros(len(Audio))
        z = np.zeros(len(Audio))
    
        for j in timing_back :
            loc = j
            scale = wide_back 
            y += norm.pdf(x,loc,scale)
            
        for j in timing_front :
            loc = j
            scale = wide_front 
            z += norm.pdf(x,loc,scale)

        carry_tone_back = np.sin(np.arange(wide_back *50)/39062.5*95*2*np.pi)
        carry_tone_front = np.sin(np.arange(wide_back *50)/39062.5*85*2*np.pi)
        a = np.argmax(carry_tone_back[1000:])
        a+=1000
        b = np.argmax(carry_tone_back [a:])
        b+=a
        
        a2 = np.argmax(carry_tone_front[1000:])
        a2+=1000
        b2 = np.argmax(carry_tone_front[a2:])
        b2+=a2

        win = 1000
        y2 = np.zeros(len(y))
        y2 += y
        for i in timing_back:
            y[i-win:i+win] = y[i-win:i+win] * carry_tone_back[b-win:b+win]
            
        win = 1000
        z2 = np.zeros(len(y))
        z2 += z
        for i in timing_front:
            z[i-win:i+win] = z[i-win:i+win] * carry_tone_front[b2-win:b2+win]
            
        self.tactile = (y + z) / (max(y+z)) * 1.4
        self.length = n_samples

    def load_into_buffer(self):
        start = time.time()
        
        audio_in_buffer = self.circuit.get_buffer('audio_in', 'w')
        tactile_in_buffer = self.circuit.get_buffer('tactile_in', 'w')



        self.circuit.set_tag('size_audio', self.length)
        self.circuit.set_tag('size_tactile', self.length)
        audio_in_buffer.write(self.audio)
        tactile_in_buffer.write(self.tactile)

    

        load=time.time()
        print('Stimuli loaded in: ',str(load-start),' seconds')
        
    def send(self):
        start = time.time()
        self.circuit.start()
        t0= time.time()
        while 1:
            if (time.time()-t0) + 0.3 > self.length / self.circuit.fs:
                break

        self.circuit.stop()
        
        end = time.time()
        print('Stimuli was sent for: ',str(end - start),' seconds')
        self.circuit.set_tag('audio_in_i', 0)
        self.circuit.set_tag('tactile_in_i', 0)
        
        


class Stimuli_Entrainment():
    def __init__(self, condition, syllable, snr, gender, circuit, fixed = False):
        
        start = time.time()
        
        #Genetal Info
        
        self.circuit = circuit
        self.type = Conditions_Entrainment[condition]['type']
        self.frequency =  Conditions_Entrainment[condition]['frequency']
        self.phase = Conditions_Entrainment[condition]['phase']
        self.Fs = 39062.5
        self.snr = snr
        self.gender = gender
        self.true_syllable = syllable
        self.false_syllable = Syllables_pair[syllable]
        self.shift = np.random.randint(int(self.Fs/2),int(self.Fs)) 
        if fixed:
            self.shift = int(self.Fs/2)
        self.random = np.random.randint(1,100)


        
        #Load Files
        path_to_noise = os.path.join(r'D:/Signal_Pierre/','ssn')
        path_to_audio = 'D:\Signal_Pierre\stimuli\Stimuli_Entrainment\Syllables'
        path_to_tactile = 'D:\Signal_Pierre\stimuli\Stimuli_Entrainment\Tactile'
        if condition == 0 or condition == 5:
            audiofile = self.true_syllable + '_' + self.gender + '_1.npy'
        else:
            audiofile = self.true_syllable + '_' + self.gender + '_' + str(condition) + '.npy'
        
        if condition == 5:
            tactilefile = 'sham_tact_' + str(self.random) + '.npy'
        else:
            tactilefile = 'tact.npy'
            
        noisefile = 'ssn_' + gender + '.npy'
        
        targetAUDIO = os.path.join(path_to_audio,audiofile)
        targetTACTILE = os.path.join(path_to_tactile,tactilefile)
        targetNOISE = os.path.join(path_to_noise,noisefile)

        
        Audio = np.load(targetAUDIO)
        Tactile = np.load(targetTACTILE)
        Noise = np.load(targetNOISE)
        
        #Scale Signals

        Tactile = Tactile / max(Tactile) * 1.5 #1.4
        Noise = Noise / RMS(Noise)
        #Noise = np.asarray(list(Noise) + list(Noise) + list(Noise) + list(Noise) + list(Noise) + list(Noise))[0:len(Audio)]
        Noise = np.asarray(list(Noise))[0:len(Audio)]
        
        Noise_Audio = AddNoisePostNorm(Audio,Noise,snr,spacing=0)
        
        #Change length Signals
        Tactile = Tactile[:int(self.Fs*1.4 + self.shift)]
        Noise = Noise[:int(self.Fs*1.4 + self.shift)]
        Noise_Audio = Noise_Audio[:int(self.Fs*1.4 + self.shift)]
        Audio = Audio[:int(self.Fs*1.4 + self.shift)]
        
        
        self.noise = Noise
        
        #Duration Info
        self.length = len(Audio)
        self.clean_audio = np.load(targetAUDIO) / 500
        self.duration = round(self.length/39062.5,2)
        
        #Handle different conditions
        self.audio = np.roll(Noise_Audio / 500, self.shift)
        
        if self.type == 'audio':
            self.tactile = np.zeros(self.length)
        else:
            self.tactile = np.roll(Tactile, self.shift)
        
        self.clean_tactile = Tactile
        
        
        self.timescale = np.arange(self.length) / self.Fs
        
        #Handle Information to save
        self.info = dict()
        self.info['Fs'] = self.Fs
        self.info['type'] = self.type
        self.info['shift'] = self.shift
        self.info['syllable'] = self.true_syllable
        self.info['score'] = 0
        self.info['random'] = self.random
        self.info['phase'] = Conditions_Entrainment[condition]['phase']
        
        print('Stimuli generated in ', str(round(time.time() - start,2)), ' seconds')
        
        
    def plot(self, *stim, new_window = True):
        start = time.time()
        if new_window:
            plt.figure()
        if len(stim) == 0:
            plt.subplot(211)
            plt.plot(self.timescale,self.audio)
            plt.subplot(212)
            plt.plot(self.timescale,self.tactile)
        else:
            for i in stim:
                if i == 'audio':
                    plt.plot(self.timescale,self.audio)
                elif i == 'tactile':
                    plt.plot(self.timescale,self.tactile/500)
                else:
                    raise Exception('Invalid Condition')
        
        
        print('Graph generated in ', str(round(time.time() - start,2)), ' seconds')
        
    def load_into_buffer(self):
        start = time.time()
        
        
        audio_in_buffer = self.circuit.get_buffer('audio_in', 'w')
        tactile_in_buffer = self.circuit.get_buffer('tactile_in', 'w')
        #stimtrack_in_buffer = self.circuit.get_buffer('stimtrack_in', 'w')
        #trigger_in_buffer = self.circuit.get_buffer('trigger_in', 'w')


        self.circuit.set_tag('size_audio', self.length)
        self.circuit.set_tag('size_tactile', self.length)
        #self.circuit.set_tag('size_stimtrack', self.length)
        #self.circuit.set_tag('size_trigger', self.length - 200)
        audio_in_buffer.write(self.audio)
        tactile_in_buffer.write(self.tactile)
        #stimtrack_in_buffer.write(self.stimtrack)
        #trigger_in_buffer.write(self.trigger)
    

        load=time.time()
        print('Stimuli loaded in: ',str(load-start),' seconds')
        
      
        
    def send(self):
        start = time.time()
        self.circuit.start()
        t0= time.time()
        while 1:
            if (time.time()-t0) - 0.2 > self.length / self.circuit.fs:
                break

        self.circuit.stop()
        
        end = time.time()
        print('Stimuli was sent for: ',str(end - start),' seconds')
        self.circuit.set_tag('audio_in_i', 0)
        self.circuit.set_tag('tactile_in_i', 0)
        #self.circuit.set_tag('stimtrack_in_i', 0)
        #self.circuit.set_tag('trigger_in_i', 0)


        
        
        


