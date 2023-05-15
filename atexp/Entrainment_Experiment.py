# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:22:44 2019

@author: phg17
"""

from tdt_function import *
from tdt_function import Conditions_Entrainment, Stimuli_Entrainment
import os.path as path
from os.path import join
from psychopy import core, visual, gui, data, event
from psychopy.tools.filetools import fromFile, toFile
import numpy, random
import numpy as np
#from utils.text_process import Keyword_Match, setup_dataframe
import pandas as pd
import os
from tdt import DSPProject
#import speech_recognition as sr
import csv
from random import randint, shuffle
import time
import itertools


wrap = 1500
snr = -10
training = False

project = DSPProject()
circuit = project.load_circuit('circuit_long_trigger.rcx', 'RX8')
fs_circuit=circuit.fs



"""Subject File Parameters"""
try: 
    expInfo = fromFile('lastParams.pickle')
except:  
    expInfo = {'observer':'training', 'subject_code':str(random.randint(1,10000000))}
expInfo['dateStr'] = data.getDateStr() 
dlg = gui.DlgFromDict(expInfo, title='Entrainment Exp', fixed=['dateStr'])
 f dlg.OK:  
    toFile('lastParams.pickle', expInfo)  
else:
    core.quit()  

fileName = expInfo['subject_code'] + '_Entrainment'
if expInfo['observer'] == 'training':
    training = True




directory = 'D:/Data_Pierre/Entrainment/'+ expInfo['subject_code'] + '_Entrainment'
os.mkdir(directory)

nb_conditions = len(Conditions_Entrainment)
conditions = np.repeat(np.arange(nb_conditions),7)
genders = ['m','f']
syllables = ['ta','da','ka','ga','pa','ba']
parameters = list(itertools.product(conditions,genders,syllables))
np.random.shuffle(parameters)
parameters = list(parameters)

np.save(join(directory,'parameters_1.npy'),parameters)


dataFile = open(join(directory,expInfo['subject_code'] + '_Entrainment' + '.csv'), 'a+',newline='')  # a simple text file with 'comma-separated-values'
fieldnames = ['Trial','Type','Phase','Score','Syllable','Gender','Shift','Random']
data_write = csv.DictWriter(dataFile, fieldnames = fieldnames)
data_write.writeheader()


"""Window and Clock"""
win = visual.Window([1950,1100], fullscr=False,allowGUI=True, monitor='testMonitor', units='pix',color=[-0.7,-0.7,-0.7])
globalClock = core.Clock()
trialClock = core.Clock()


"""Starting Message"""
message1 = visual.TextStim(win, pos=[0,0], height = 50, text='Welcome to this Experiment', wrapWidth=wrap)
message2 = visual.TextStim(win, pos=[0,-350], height = 45, text='[Hit a key to continue]', wrapWidth=wrap)
message1.draw()
message2.draw()
win.flip()
event.waitKeys()


"""Starting Message"""
message1 = visual.TextStim(win, pos=[0,0], height = 50, text='You will be asked to tell which syllable you recognized using the left and right keys', wrapWidth=wrap)
message2 = visual.TextStim(win, pos=[0,-350], height = 45, text='[Hit a key to continue]', wrapWidth=wrap)
message1.draw()
message2.draw()
win.flip()
event.waitKeys()

if training:
    parameters = parameters[:50]
    expInfo['subject_code'] = 'training'
    
    
trials = np.arange(len(parameters))

for trial in trials:
    
    param = parameters[trial]

    stimuli = Stimuli_Entrainment(param[0],param[2],snr,param[1],circuit)
    
    print(stimuli.info)

    stimuli.load_into_buffer()

    fixation_cross_1 = visual.Line(win,start=[0,-250],end=[0,250],lineWidth=10)
    fixation_cross_2 = visual.Line(win,start=[-250,0],end=[250,0],lineWidth=10)
    fixation_cross_1.draw()
    fixation_cross_2.draw()
    win.flip()
    
    stimuli.send()
    '''Send Stimuli'''

    
    '''Questions'''
    true = stimuli.true_syllable
    false = stimuli.false_syllable
    right_true = np.random.randint(0,2)
    if right_true:
        RIGHT = visual.TextStim(win, pos = [500,0], height = 80, text = true, wrapWidth=wrap)
        LEFT = visual.TextStim(win, pos = [-500,0], height = 80, text = false, wrapWidth=wrap)
    else:
        RIGHT = visual.TextStim(win, pos = [500,0], height = 80, text = false, wrapWidth=wrap)
        LEFT = visual.TextStim(win, pos = [-500,0], height = 80, text = true, wrapWidth=wrap)
    keyl = visual.TextStim(win, pos = [-500,-300], height = 40, text = 'left key', wrapWidth=wrap)
    keyr = visual.TextStim(win, pos = [500,-300], height = 40, text = 'right key', wrapWidth=wrap)
    keyl.draw()
    keyr.draw()
    RIGHT.draw()
    LEFT.draw()
    win.flip()
    response = None
    while response == None:
        allKeys=event.waitKeys()
        for thisKey in allKeys:
            if thisKey == 'right':
                response = 'right'
            elif thisKey == 'left':
                response = 'left'
            elif thisKey in ['q', 'escape']:
                core.quit()
            else:
                #raise Exception('problem with Keyboard dude')
                print('incorrect input')
        event.clearEvents()
    if (right_true and response == 'right') or (not(right_true) and response == 'left'):
        stimuli.info['score'] = 1
    else:
        stimuli.info['score'] = 0


    data_write.writerow({'Trial':trial,'Type':stimuli.info['type'],'Phase':stimuli.info['phase'],'Score':stimuli.info['score'],'Syllable':stimuli.info['syllable'],'Gender':stimuli.gender,'Shift':stimuli.info['shift'],'Random':stimuli.info['random']})

    if trial % 30 == 0 and trial!=0:
        perc = int(trial/len(trials)*100)
        message1 = visual.TextStim(win, pos = [0,0], height = 80, text = 'Take a Break', wrapWidth=wrap)
        message2 = visual.TextStim(win, pos=[0,-350], height = 45, text='[Hit a key to continue]', wrapWidth=wrap)
        message3 = visual.TextStim(win, pos=[0,-150], height = 80, text= str(perc) + '% Done!', wrapWidth=wrap)
        
        message1.draw()
        message2.draw()
        message3.draw()
        win.flip()
        event.waitKeys()
    
dataFile.close()



win.close()
core.quit()


