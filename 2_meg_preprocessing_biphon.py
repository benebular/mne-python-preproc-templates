### Preprocessing script for BIPHON, adapted from Laura Gwilliams leg5@nyu.edu
## Author: Ben Lang, ben.lang@nyu.edu

## BEFORE STARTING: Did you kit2fiff and coreg?
# change your mne coreg dir with export SUBJECTS_DIR=(insert file path here)

import numpy as np
import pandas as pd
import os
import os.path as op
# import wx
import mne
import sys
from mne.io import read_raw_fif
from mne.preprocessing.ica import read_ica
from mne import (pick_types, find_events, Epochs, Evoked, compute_covariance,
                 write_cov, read_cov, setup_source_space, write_source_spaces, make_bem_model, make_bem_solution, make_bem_solution, make_forward_solution,
                 write_forward_solution, read_forward_solution, write_bem_solution, convert_forward_solution, read_epochs, grade_to_vertices,
                 compute_source_morph, compute_morph_matrix, read_source_estimate, convert_forward_solution)
# from mne import (pick_types, find_events, Epochs, Evoked, compute_covariance,
#                  write_cov, read_cov, setup_source_space, write_source_spaces, make_bem_model, make_bem_solution, make_bem_solution, make_forward_solution,
#                  write_forward_solution, read_forward_solution, write_bem_solution, convert_forward_solution, read_epochs, grade_to_vertices,
#                  read_source_estimate)
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, apply_inverse, write_inverse_operator, read_inverse_operator
from mne.preprocessing import ICA
from mne.viz import plot_source_estimates
# import matplotlib.pyplot as plt
mne.set_log_level(verbose='INFO')
np.set_printoptions(threshold=sys.maxsize)

# from logfile_parse import df as trial_info
# %gui qt

# params
subjects = ['A0280','A0318','A0392','A0396','A0416','A0417']
filt_l = 0
filt_h = 60
tmin = -0.2
tmax = 1.5

for subject in subjects:

    # meg_dir = '/Volumes/MEG/NYUAD-Lab-Server/DataAnalysis/Ben/MEG/biphon/meg/%s/'%(subject) # change to local meg folder
    meg_dir = '/Volumes/bin_battuta/biphon/meg/%s/'%(subject)
    # meg_dir = '/scratch/bcl267/DataAnalysis/%s/'%(subject) # change to local meg folder

    # for plotting
    os.environ["SUBJECTS_DIR"] = '/Volumes/bin_battuta/biphon/mri/' # change to local mri folder

    # file names, MEG preproc
    trial_info_fname = meg_dir + subject + '_log_cleaned.csv'
    raw_fname = meg_dir + subject+ '_biphon-raw.fif'
    ica_fname = meg_dir + subject+ '_biphon_ica1-ica.fif'
    ica_raw_fname = meg_dir + subject+ '_ica_biphon-raw.fif' # applied ica to raw
    filt_raw_fname = meg_dir + subject + '_filt_biphon-raw.fif'
    # ica_fname = meg_dir + subject+ '_biphon_ica1-ica-2.fif'
    # ica_raw_fname = meg_dir + subject+ '_ica_biphon-raw-2.fif' # applied ica to raw
    pickled_rej_A_fname = meg_dir + subject+ '_rejfile_A.pickled' # rejected epochs after ica
    pickled_rej_B_fname = meg_dir + subject+ '_rejfile_B.pickled' # rejected epochs after ica
    pickled_rej_X_fname = meg_dir + subject+ '_rejfile_X.pickled' # rejected epochs after ica
    epochs_A_fname = meg_dir + subject+ '_biphon_A-epo.fif'
    epochs_B_fname = meg_dir + subject+ '_biphon_B-epo.fif'
    epochs_X_fname = meg_dir + subject+ '_biphon_X-epo.fif'
    epochs_A_cropped_fname = meg_dir + subject + '_biphon_A_cropped-epo.fif'
    epochs_B_cropped_fname = meg_dir + subject + '_biphon_B_cropped-epo.fif'
    epochs_X_cropped_fname = meg_dir + subject + '_biphon_X_cropped-epo.fif'
    epochs_cropped_fname = meg_dir + subject + '_biphon_cropped-epo.fif'
    cov_fname = meg_dir + subject+ '_biphon-cov.fif'
    evoked_A_fname = meg_dir + subject+ '_biphon_A-evoked-ave.fif'
    evoked_B_fname = meg_dir + subject+ '_biphon_B-evoked-ave.fif'
    evoked_X_fname = meg_dir + subject+ '_biphon_X-evoked-ave.fif'


    # filenames, MRI/source space creation
    mri_dir = '/Volumes/bin_battuta/biphon/mri/%s/bem/'%subject # subject's bem folder
    fwd_fname = os.path.join(meg_dir, '%s_biphon-fwd.fif'%subject)
    inv_fname = os.path.join(meg_dir, '%s_biphon-inv.fif'%subject)
    bem_fname = os.path.join(mri_dir, '%s-inner_skull-bem-sol.fif'%subject)
    src_fname = os.path.join(mri_dir, '%s-ico-4-src.fif'%subject)
    trans_fname = os.path.join(mri_dir, '%s-trans.fif'%subject)

    evo_whi_fname = os.path.join(meg_dir, '%s-whitened.jpg'%subject)

    #--------------------------------------------
    # START PREPROC

    # if the ica-clean raw exists, load it
    if op.isfile(ica_raw_fname):
        print ('Reading in raw post-ICA for %s.' %(subject)) # number of events
        raw = read_raw_fif(ica_raw_fname, preload=True)

        raw2 = raw.copy()
        raw2.info['bads'] = []
        events = find_events(raw2, stim_channel='STI 014', min_duration=0.002)
        epochs = Epochs(raw2, events=events, event_id=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], tmin=-0.2, tmax=1.5, baseline=None).average() # pick an Event ID
        epochs_plt = epochs.plot()
        epochs_plt.savefig(meg_dir + '%s_rawpostica.png'%subject)
        del raw2, epochs, events

    # else, make it
    else:
        # step 1- concatenate data for each block
        raw = read_raw_fif(raw_fname, preload=True)
        print ('Reading in raw data for %s from %s.' %(subject, raw_fname)) # number of events

        # plot raw data to check for magnetic noise etc
        raw2 = raw.copy()
        raw2.info['bads'] = []
        events = find_events(raw2, stim_channel='STI 014', min_duration=0.002)
        epochs = Epochs(raw2, events=events, event_id=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], tmin=-0.2, tmax=1.5, baseline=None).average().plot() # pick an Event ID
        del raw2, epochs, events

        # step 2- remove bad channels
        print (raw.info['bads'])  # check if any bad channels have been specified already
        raw.plot()  # visualise bad channels
        raw.info['bads'] = ['list_of_bad_channels']
        # interpolate bads and reset so that we have same number of channels for all blocks/subjects
        raw.interpolate_bads(reset_bads=True)
        print ('Interpolating bad channels...') # number of events

        raw.save(raw_fname, overwrite=True)  # overwrite w/ bad channel info/interpolated bads

        # step 3- apply ICA to the conjoint data
        picks = pick_types(raw.info, meg=True, exclude='bads')
        ica = ICA(n_components=0.95, method='fastica')

        # get ica components
        ica.exclude = []
        ica.fit(raw, picks=picks, decim=5)
        ica.save(ica_fname)  # save solution

        # view components and make rejections
        ica.plot_sources(raw)

        # if anything quits at this points, reload ICA:
        # ica = read_ica(ica_fname)

        # apply ica to raw
        ica.apply(raw)

        ## SANITY CHECK: How does the data look after ICA?
        # show some frontal channels to clearly illustrate the artifact removal
        orig_raw = read_raw_fif(raw_fname, preload=True)
        orig_raw.plot()
        raw.plot()
        del orig_raw

        #plot post-ICA
        raw3 = raw.copy()
        raw3.info['bads'] = []
        events3 = find_events(raw3, stim_channel='STI 014', min_duration=0.002)
        epochs3 = Epochs(raw3, events=events, event_id=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], tmin=-0.2, tmax=1.5, baseline=None).average().plot() # pick an Event ID
        del raw3, epochs3, events3

        # save result ica file as raw
        raw.save(ica_raw_fname, overwrite=True)

        # step 4- filter
        print ('Filtering raw data (post-ICA) for %s.' %(subject)) # number of events
        raw = raw.filter(filt_l, filt_h)
        raw.save(filt_raw_fname)

    # step 5- make epochs
    event_id = dict(i1_A=10, i2_A=11, u1_A=12, u2_A=13, ah1_A=16, ah2_A=17, ae1_A=14, ae2_A=15, yih1_A=18, yih2_A=19, y1_A=20, y2_A=21, ob1_A=22, ob2_A=23, uu1_A=24, uu2_A=25,
                    i1_B=26, i2_B=27, u1_B=28, u2_B=29, ah1_B=30, ah2_B=31, ae1_B=32, ae2_B=33, yih1_B=34, yih2_B=35, y1_B=36, y2_B=37, ob1_B=38, ob2_B=39, uu1_B=40, uu2_B=41,
                    i1_X=42, i2_X=43, u1_X=44, u2_X=45, ah1_X=46, ah2_X=47, ae1_X=48, ae2_X=49, yih1_X=50, yih2_X=51, y1_X=52, y2_X=53, ob1_X=54, ob2_X=55, uu1_X=56, uu2_X=57)
    print ('Finding events for %s...'%subject)
    events = find_events(raw, stim_channel='STI 014', min_duration=0.002)  # the output of this is a 3 x n_trial np array
    print ('%s events found in raw.' %len(events)) # number of events

    # audio delay, adds however many you need to account for delay to first column of each NP array entry
    print ("Adding audio delay...")
    evnt = []
    for i in range(len(events)):
        events[i][0] += 50
        evnt.append(events[i])
    events = np.array(evnt)

    ###

    # import csv
    # with open('A0280_events.csv', 'w') as file:
    #     w = csv.writer(file)
    #     w.writerow(id_event)

    # data = pd.read_excel('/Volumes/bin_battuta/biphon/meg/A0280/A0280_events.xls')
    # new_events = data.events.tolist()
    # for x in range(len(new_events)):
    #     events[x][2] = new_events[x]

    # if events are funky and longer because responses are included, use below to index by trigger code in third column in np array
    # evnt = []
    # for i in range(len(events)):
    #     if events[i][2] == 1:
    #         evnt.append(events[i])
    # events = np.array(evnt)
    #
    # # replace triggers for nonnative
    # trial_info = pd.read_csv(trial_info_fname)
    #
    # for i in range(len(trial_info)):
    #         if trial_info.iloc[i,6] == 0:
    #             events[i][2] = 7

    # epochs = Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=0.5, baseline=None, preload=True)
    # print ('%s events found in raw.' %len(epochs)) # number of events

    # note: you may want to add some decimation here
    # 'i1_A', 'i2_A', 'u1_A', 'u2_A', 'ah1_A', 'ah2_A', 'ae1_A', 'ae2_A', 'yih1_A', 'yih2_A', 'y1_A', 'y2_A', 'ob1_A', 'ob2_A', 'uu1_A', 'uu2_A'
    print('Creating separate epochs for A, B, and X for %s...'%subject)
    epochs_A = Epochs(raw, events, event_id=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], tmin=-0.2, tmax=1.5, baseline=(-0.2,0), preload=True)
    epochs_B = Epochs(raw, events, event_id=[26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41], tmin=-0.7, tmax=0.5, baseline=(-0.7,-0.5), preload=True)
    epochs_X = Epochs(raw, events, event_id=[42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57], tmin=-1.2, tmax=0.5, baseline=(-1.2,-1.0), preload=True)
    trial_info = pd.read_csv(trial_info_fname)
    trial_info_A = trial_info[trial_info['position']==1]
    trial_info_B = trial_info[trial_info['position']==2]
    trial_info_X = trial_info[trial_info['position']==3]
    print('Found %s epochs.'%(len(epochs_A)+len(epochs_B)+len(epochs_X)))

    epochs_A.save(epochs_A_fname, overwrite=True)
    epochs_B.save(epochs_B_fname, overwrite=True)
    epochs_X.save(epochs_X_fname, overwrite=True)

    del raw, epochs_A, epochs_B, epochs_X
