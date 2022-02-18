#!/usr/bin/env python
# -*- coding: utf-8 -*

# ea84@nyu.edu
# univariate analysis of shepard data
# goal: loop to create ICA for data with bad channels already removed

# adapted by bcl267@nyu.edu for biphon project

import numpy as np
import os
import time
import eelbrain
import os.path as op
from mne.io import read_raw_fif
from mne.preprocessing.ica import read_ica
from mne import (pick_types, find_events, Epochs, Evoked, compute_covariance,
                 write_cov, read_cov, setup_source_space, make_forward_solution,
                 read_forward_solution, convert_forward_solution)
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, apply_inverse
from mne.preprocessing import (ICA, read_ica)
from sklearn.decomposition import FastICA
import pandas as pd


subjects = ['A0280','A0318','A0392','A0396','A0416','A0417']
# params
filt_l = 1  # same as aquisition
filt_h = 60
tmin = -0.2
tmax = 1.5
decim = 2
reject = dict(mag=4e-12)
random_state = 42
max_iter = 10000

# (6 subjects params: decim = 2, reject mag 4e-12, random state = 42, max iter = 10k) as of Mar 2 2020

for subject in subjects:
    print (subject)
    meg_dir = '/Volumes/bin_battuta/biphon/meg/%s/'%(subject)
    raw_fname = meg_dir + subject+ '_biphon-raw.fif'
    ica_fname = meg_dir + subject+ '_biphon_ica1-ica.fif'
    ica_raw_fname = meg_dir + subject+ '_ica_biphon-raw.fif' # applied ica to raw

    print ("Reading raw file for %s..."%subject)
    t = time.time()
    raw = read_raw_fif(raw_fname, preload=True)
    elapsed_readraw = time.time() - t
    print(elapsed_readraw)
    print ("Filtering data...")
    raw = raw.filter(filt_l, filt_h)

    # print ("Finding events...")
    # events = find_events(raw, stim_channel='STI 014', min_duration=0.002)
    # print ('%s events found in raw.' %len(events)) # the output of this is a 3 x n_trial np array
    #
    # print ("Adding audio delay...")
    # evnt = [] # the below loop adds 50ms to the time in each trial to account for auditory delay
    # for i in range(len(events)):
    #     events[i][0] += 50
    #     evnt.append(events[i])
    # events = np.array(evnt)
    #
    # # create an event dictionary based on conditions, in this case, each vowel presented
    # event_id = dict(i1_A=10, i2_A=11, u1_A=12, u2_A=13, ah1_A=16, ah2_A=17, ae1_A=14, ae2_A=15, yih1_A=18, yih2_A=19, y1_A=20, y2_A=21, ob1_A=22, ob2_A=23, uu1_A=24, uu2_A=25,
    #                 i1_B=26, i2_B=27, u1_B=28, u2_B=29, ah1_B=30, ah2_B=31, ae1_B=32, ae2_B=33, yih1_B=34, yih2_B=35, y1_B=36, y2_B=37, ob1_B=38, ob2_B=39, uu1_B=40, uu2_B=41,
    #                 i1_X=42, i2_X=43, u1_X=44, u2_X=45, ah1_X=46, ah2_X=47, ae1_X=48, ae2_X=49, yih1_X=50, yih2_X=51, y1_X=52, y2_X=53, ob1_X=54, ob2_X=55, uu1_X=56, uu2_X=57)
    #
    # print ("Epoching data...")
    # epochs = Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True)
    # print ('%s epochs found in raw.' %len(epochs)) # the output of this is a 3 x n_trial np array


    print ("Creating ICA object...")
    # apply ICA to the conjoint data
    picks = pick_types(raw.info, meg=True, exclude='bads')
    ica = ICA(n_components=0.95,method='fastica', max_iter=max_iter, random_state=random_state)

    print ("Fitting ICA...")
    # get ica components
    ica.exclude = []
    t = time.time()
    ica.fit(raw, picks=picks, decim=decim, reject=reject)
    elapsed_ica = time.time() - t
    print("ICA fit elapsed time in seconds: %s" %elapsed_ica)


    print ("Saving ICA solution...")
    ica.save(ica_fname)  # save solution

#__________________________________________________________
subject = 'A0280' # change subject and follow steps below to apply each individual ICA and select bad compoenents manually

meg_dir = '/Volumes/bin_battuta/biphon/meg/%s/'%(subject)
raw_fname = meg_dir + subject+ '_biphon-raw.fif'
ica_fname = meg_dir + subject+ '_biphon_ica1-ica.fif'
ica_raw_fname = meg_dir + subject+ '_ica_biphon-raw.fif' # applied ica to raw

# params
filt_l = 1  # same as aquisition
filt_h = 60
tmin = -0.2
tmax = 1.5

print ("Reading raw file...")
raw = read_raw_fif(raw_fname, preload=True)

# print (raw.info['bads'])  # check if any bad channels have been specified already
# raw.plot()  # visualise bad channels
# raw.info['bads'] = ['list_of_bad_channels']
# # interpolate bads and reset so that we have same number of channels for all blocks/subjects
# raw.interpolate_bads(reset_bads=True)
# print ('Interpolating bad channels...') # number of events

print ("Filtering data...")
raw = raw.filter(filt_l, filt_h)

## visualize post-filter raw data
# raw2 = raw.copy()
# raw2.info['bads'] = []
# events = find_events(raw2, stim_channel='STI 014', min_duration=0.002)
# epochs = Epochs(raw2, events=events, event_id=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], tmin=-0.2, tmax=1.5, baseline=None).average().plot() # pick an Event ID
# del raw2, epochs, events

print ("Reading ICA...")
ica = read_ica(ica_fname)
ica.plot_sources(raw)

ica.apply(raw)
raw.save(ica_raw_fname, overwrite=True)

## visualize post-ICA raw data
# raw3 = raw.copy()
# raw3.info['bads'] = []
# events3 = find_events(raw3, stim_channel='STI 014', min_duration=0.002)
# epochs3 = Epochs(raw3, events=events, event_id=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], tmin=-0.2, tmax=1.5, baseline=None).average().plot() # pick an Event ID
# del raw3, epochs3, events3
