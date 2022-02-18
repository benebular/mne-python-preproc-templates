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
# from mne import (pick_types, find_events, Epochs, Evoked, compute_covariance,
#                  write_cov, read_cov, setup_source_space, write_source_spaces, make_bem_model, make_bem_solution, make_bem_solution, make_forward_solution,
#                  write_forward_solution, read_forward_solution, write_bem_solution, convert_forward_solution, read_epochs, grade_to_vertices,
#                  compute_source_morph, compute_morph_matrix, read_source_estimate, convert_forward_solution)
from mne import (pick_types, find_events, Epochs, Evoked, compute_covariance,
                 write_cov, read_cov, setup_source_space, write_source_spaces, make_bem_model, make_bem_solution, make_bem_solution, make_forward_solution,
                 write_forward_solution, read_forward_solution, write_bem_solution, convert_forward_solution, read_epochs, grade_to_vertices,
                 read_source_estimate)
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, apply_inverse, write_inverse_operator, read_inverse_operator
from mne.preprocessing import ICA
from mne.viz import plot_source_estimates
import matplotlib.pyplot as plt
mne.set_log_level(verbose='INFO')
np.set_printoptions(threshold=sys.maxsize)

# from logfile_parse import df as trial_info
# %gui qt

# params
subjects = ['A0280','A0318','A0392','A0396','A0416','A0417']
filt_l = 0
filt_h = 40
tmin = -0.2
tmax = 1.5

for subject in subjects:
    # meg_dir = '/Volumes/MEG/NYUAD-Lab-Server/DataAnalysis/Ben/MEG/biphon/meg/%s/'%(subject) # change to local meg folder
    dec_dir = '/Volumes/hecate/biphon/dec/%s/'%(subject)
    # meg_dir = '/scratch/bcl267/DataAnalysis/%s/'%(subject) # change to local meg folder

    # for plotting
    os.environ["SUBJECTS_DIR"] = '/Volumes/hecate/biphon/mri/' # change to local mri folder

    # file names, MEG preproc
    trial_info_fname = dec_dir + subject + '_log_cleaned.csv'
    raw_fname = dec_dir + subject+ '_biphon-raw.fif'
    ica_fname = dec_dir + subject+ '_biphon_ica1-ica.fif'
    ica_raw_fname = dec_dir + subject+ '_ica_biphon-raw.fif' # applied ica to raw
    filt_raw_fname = dec_dir + subject + '_filt_biphon-raw.fif'
    pickled_rej_fname = dec_dir + subject+ '_rejfile.pickled' # rejected epochs after ica
    epochs_fname = dec_dir + subject+ '_biphon-epo.fif'
    epochs_cropped_fname = dec_dir + subject + '_biphon_cropped-epo.fif'
    epochs_baseline_cropped_fname = dec_dir + subject + '_biphon_baseline_cropped-epo.fif'
    cov_fname = dec_dir + subject+ '_biphon-cov.fif'
    evoked_fname = dec_dir + subject+ '_biphon-evoked-ave.fif'

    # filenames, MRI/source space creation
    mri_dir = '/Volumes/hecate/biphon/mri/%s/bem/'%subject # subject's bem folder
    fwd_fname = os.path.join(dec_dir, '%s_biphon-fwd.fif'%subject)
    inv_fname = os.path.join(dec_dir, '%s_biphon-inv.fif'%subject)
    bem_fname = os.path.join(mri_dir, '%s-inner_skull-bem-sol.fif'%subject)
    src_fname = os.path.join(mri_dir, '%s-ico-4-src.fif'%subject)
    trans_fname = os.path.join(mri_dir, '%s-trans.fif'%subject)

    evo_whi_fname = os.path.join(dec_dir, '%s-whitened.jpg'%subject)

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
        epochs_plt.savefig(dec_dir + '%s_rawpostica.png'%subject)
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

    epochs = Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=0.5, baseline=None, preload=True)
    print ('%s epochs found in raw.' %len(epochs)) # number of events
    epochs.save(epochs_fname, overwrite=True)


# epochs_A.save(epochs_A_fname, overwrite=True)
# epochs_B.save(epochs_B_fname, overwrite=True)

# note: you may want to add some decimation here
# 'i1_A', 'i2_A', 'u1_A', 'u2_A', 'ah1_A', 'ah2_A', 'ae1_A', 'ae2_A', 'yih1_A', 'yih2_A', 'y1_A', 'y2_A', 'ob1_A', 'ob2_A', 'uu1_A', 'uu2_A'
# print('Creating separate epochs for A, B, and X for %s'%subject)
# epochs_A = Epochs(raw, events, event_id=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], tmin=-0.2, tmax=1.5, baseline=(-0.2,0), preload=True)
# epochs_B = Epochs(raw, events, event_id=[26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41], tmin=-0.7, tmax=0.5, baseline=(-0.7,-0.5), preload=True)
# epochs_X = Epochs(raw, events, event_id=[42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57], tmin=-1.2, tmax=0.5, baseline=(-1.2,-1.0), preload=True)


##____________rejections_______________##
import numpy as np
import pandas as pd
import os
import os.path as op
import mne
import eelbrain

subject = 'A0417'
dec_dir = '/Volumes/hecate/biphon/dec/%s/'%(subject)
os.environ["SUBJECTS_DIR"] = '/Volumes/hecate/biphon/mri/' # change to local mri folder
pickled_rej_fname = dec_dir + subject+ '_rejfile.pickled' # rejected epochs after ica
epochs_fname = dec_dir + subject+ '_biphon-epo.fif'
trial_info_fname = dec_dir + subject + '_log_cleaned.csv'

trial_info = pd.read_csv(trial_info_fname)

# if not already modified via regex_biphon_logs.py, use section below to make dummy groups
# import re
# text = trial_info
# trial_info['vowel_id'] = trial_info['vowel_id'].astype('str')
# trial_info['vowel_iso'] = trial_info['vowel_id'].str.extract(r'(.*)\d')
# trial_info_dummies = pd.get_dummies(trial_info['vowel_iso'])
# trial_info = pd.concat([trial_info, trial_info_dummies], axis=1)
# di_1={"i":"1","y":"1","yih":"1"}
# di_2={"u":"1","uu":"1","ob":"1"}
# di_3={"ae":"1","ah":"1"}
# trial_info['vowel_grp1']=trial_info['vowel_iso'].map(di_1).fillna("0")
# trial_info['vowel_grp2']=trial_info['vowel_iso'].map(di_2).fillna("0")
# trial_info['vowel_grp3']=trial_info['vowel_iso'].map(di_3).fillna("0")
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#         print(trial_info)
# def f(var):
#     if isinstance(var, pd.DataFrame):
#         print("It's a DataFrame!")
# f(trial_info)

# trial_info_baseline = pd.read_csv(trial_info_fname)

# trial_info_A = trial_info[trial_info['vowel_position']==1]
# trial_info_B = trial_info[trial_info['vowel_position']==2]
# trial_info_X = trial_info[trial_info['vowel_position']==3]
# print('Found %s epochs.'%(len(epochs_A)+len(epochs_B)+len(epochs_X)))

# epochs_A.save(epochs_A_fname, overwrite=True)
# epochs_B.save(epochs_B_fname, overwrite=True)
# epochs_X.save(epochs_X_fname, overwrite=True)

# del raw, epochs_A, epochs_B, epochs_X

# step 6- reject epochs based on threshold (2e-12)
# opens the gui, "mark" is to mark in red the channel closest to the eyes
from eelbrain.gui import select_epochs
from eelbrain.load import unpickle
# epochs_A = mne.read_epochs(epochs_A_fname)
epochs = mne.read_epochs(epochs_fname)

if op.isfile(pickled_rej_fname):
    rejfile = unpickle(pickled_rej_fname)
else:
    select_epochs(epochs, vlim=2e-12, mark=['MEG 087','MEG 130'])
    # This reminds you how to name the file and also stops the loop until you press enter
    raw_input('NOTE: Save as subj_rejfile.pickled. \nPress enter when you are done rejecting epochs in the GUI...')
    rejfile = unpickle(pickled_rej_fname)

# create a mask to reject the bad epochs
rejs = rejfile['accept'].x

# mask epochs and info
epochs = epochs[rejs]
trial_info = trial_info[rejs]

log_mask = np.where(trial_info.reject==1, True, False)
epochs = epochs[log_mask]
trial_info = trial_info[log_mask]


# set metadata: allows you to specify more complex info about events,
# can use pandas-style queries to access subsets of data
epochs.metadata = trial_info

# SANITY CHECK!!:
assert(len(epochs.events) == len(trial_info))

epochs.save(epochs_fname, overwrite=True)

trial_info.to_csv(dec_dir+'%s_biphon_rej_trialinfo.csv'%(subject))


##Baseline
# if op.isfile(pickled_rej_baseline_fname):
#     rejfile_baseline = unpickle(pickled_rej_baseline_fname)
# else:
#     select_epochs(epochs_baseline, vlim=2e-12, mark=['MEG 087','MEG 130'])
#     # This reminds you how to name the file and also stops the loop until you press enter
#     raw_input('NOTE: Save as subj_rejfile.pickled. \nPress enter when you are done rejecting epochs in the GUI...')
#     rejfile_baseline = unpickle(pickled_rej_baseline_fname)
#
# # create a mask to reject the bad epochs
# rejs_baseline = rejfile_baseline['accept'].x
#
# # mask epochs and info
# epochs_baseline = epochs_baseline[rejs_baseline]
# trial_info_baseline = trial_info_baseline[rejs_baseline]
#
# # set metadata: allows you to specify more complex info about events,
# # can use pandas-style queries to access subsets of data
#
# epochs_baseline.metadata = trial_info_baseline
# epochs_baseline.save(epochs_baselinefname)
#
# # SANITY CHECK!!:
# assert(len(epochs_baseline.events) == len(trial_info_baseline))
# trial_info_baseline.to_csv(meg_dir+'%s_biphon_rej_baseline_trialinfo.csv'%(subject))
#

# del epochs_A


#### B TRIALS
# epochs_B = mne.read_epochs(epochs_B_fname)
# if op.isfile(pickled_rej_B_fname):
#     rejfile = eelbrain.load.unpickle(pickled_rej_B_fname)
# else:
#     eelbrain.gui.select_epochs(epochs_B, vlim=2e-12, mark=['MEG 087','MEG 130'])
#     # This reminds you how to name the file and also stops the loop until you press enter
#     raw_input('NOTE: Save as subj_rejfile.pickled. \nPress enter when you are done rejecting epochs in the GUI...')
#     rejfile = eelbrain.load.unpickle(pickled_rej_B_fname)
#
# # create a mask to reject the bad epochs
# rejs_B = rejfile['accept'].x
#
# # mask epochs and info
# epochs_B = epochs_B[rejs_B]
# trial_info_B = trial_info_B[rejs_B]
#
# # set metadata: allows you to specify more complex info about events,
# # can use pandas-style queries to access subsets of data
#
# epochs_B.metadata = trial_info_B
# epochs_B.save(epochs_B_fname)
#
# # SANITY CHECK!!:
# assert(len(epochs_B.events) == len(trial_info_B))
# trial_info_B.to_csv(meg_dir+'%s_biphon_B_rej_trialinfo.csv'%(subject))
#
# del epochs_B
#
# ##### X TRIALS
# epochs_X = mne.read_epochs(epochs_X_fname)
# if op.isfile(pickled_rej_X_fname):
#     rejfile = eelbrain.load.unpickle(pickled_rej_X_fname)
# else:
#     eelbrain.gui.select_epochs(epochs_X, vlim=2e-12, mark=['MEG 087','MEG 130'])
#     # This reminds you how to name the file and also stops the loop until you press enter
#     raw_input('NOTE: Save as subj_rejfile.pickled. \nPress enter when you are done rejecting epochs in the GUI...')
#     rejfile = eelbrain.load.unpickle(pickled_rej_X_fname)
#
# # create a mask to reject the bad epochs
# rejs_X = rejfile['accept'].x
#
# # mask epochs and info
# epochs_X = epochs_X[rejs_X]
# trial_info_X = trial_info_X[rejs_X]
#
# # set metadata: allows you to specify more complex info about events,
# # can use pandas-style queries to access subsets of data
#
# epochs_X.metadata = trial_info_X
# epochs_X.save(epochs_X_fname)
#
# # SANITY CHECK!!:
# assert(len(epochs_X.events) == len(trial_info_X))
# trial_info_X.to_csv(meg_dir+'%s_biphon_X_rej_trialinfo.csv'%(subject))
#
# del epochs_X
#
# ## make a big csv with all epochs
# new_trial_info = pd.concat([trial_info_A,trial_info_B,trial_info_X])
# new_trial_info.to_csv(meg_dir+'%s_biphon_rej_trialinfo.csv'%(subject))


# epochs = mne.read_epochs(epochs_fname)
# trial_info = pd.read_csv('/Volumes/bin_battuta/biphon/meg/A0396/A0396_biphon_rej_trialinfo.csv')
# epochs.metadata = trial_info
# epochs = epochs.crop(-0.2,0)
# epochs_A = epochs['vowel_position == 1']
# epochs_A.save(meg_dir + subject + '_biphon_cropped_baseline-epo.fif')
# epochs_A = epochs_A.crop(0,0.5)
# epochs = mne.concatenate_epochs([epochs_A,epochs_B,epochs_X])


#-------------------------------------------------------------------------------
# slice out the epochs from A because they are the entire trial period
# epochs_A = epochs['vowel_position == 1']
# epochs_B = epochs['vowel_position == 2']
# epochs_X = epochs['vowel_position == 3']
#
#
# # step 7- make noise covariance matrix
# if not op.isfile(cov_fname):
#     print ('Computing noise covariance for %s.' %(subject)) # number of events
#     noise_cov = compute_covariance(epochs_A, tmax=0., method=['shrunk'])
#     write_cov(cov_fname, noise_cov)
# else:
#     noise_cov = read_cov(cov_fname)
#
# #check noise whitening
# evoked = epochs_A.average()
# evo_whi = evoked.plot_white(noise_cov, time_unit='s')
# evo_whi.savefig(evo_whi_fname, dpi=150)
# del evoked
#
# # if using native MRI, need to make_bem_model
# if not op.isfile(bem_fname):
#     print ('Making and writing BEM for %s.' %(subject)) # number of events
#     surfaces = make_bem_model(subject, ico=4, conductivity=([0.3]), verbose=True)
#     bem = make_bem_solution(surfaces)
#     write_bem_solution(bem_fname, bem)
#
# # step 8- make forward solution
# if not op.isfile(fwd_fname):
#     if not op.isfile(src_fname):
#         print ('Making and writing forward solution for %s.' %(subject)) # number of events
#         src = setup_source_space(subject, spacing='ico4')
#         write_source_spaces(src_fname, src, overwrite=True, verbose=None)
#         fwd = make_forward_solution(epochs_A.info, trans_fname, src, bem_fname,
#                                  meg=True, ignore_ref=True)
#         fwd = convert_forward_solution(fwd,force_fixed=True)
#         write_forward_solution(fwd_fname, fwd, overwrite=True, verbose=None)
#     else:
#         src = src_fname
#         fwd = make_forward_solution(epochs_A.info, trans_fname, src, bem_fname,
#                                  meg=True, ignore_ref=True)
#         fwd = convert_forward_solution(fwd,force_fixed=True)
#         write_forward_solution(fwd_fname, fwd, overwrite=True, verbose=None)
#
# else:
#     fwd = read_forward_solution(fwd_fname)
#     fwd = convert_forward_solution(fwd,force_fixed=True) #missing from previous exps... FUCKING GREAT!!!
#
# # step 9- make inverse solution for epochs
# if not op.isfile(inv_fname):
#     print ('Making and writing inverse operator for %s.' %(subject)) # number of events
#     inverse_operator = make_inverse_operator(epochs_A.info, fwd, noise_cov,
#                                          loose=0, depth=None, fixed=True)
#     write_inverse_operator(inv_fname, inverse_operator, verbose=None)
# else:
#     inverse_operator = read_inverse_operator(inv_fname, verbose=None)
#
# epochs_A_baseline = epochs_A.crop(-0.2,0)
# epochs_B_baseline = epochs_B.crop(-0.2,0)
# epochs_X_baseline = epochs_X.crop(-0.2,0)
# epochs_A = epochs['vowel_position == 1']
# epochs_B = epochs['vowel_position == 2']
# epochs_X = epochs['vowel_position == 3']
# epochs_A = epochs_A.crop(0,0.5)
# epochs_B = epochs_B.crop(0,0.5)
# epochs_X = epochs_X.crop(0,0.5)
# epochs_baseline = mne.concatenate_epochs([epochs_A_baseline,epochs_B_baseline,epochs_X_baseline])
# epochs_baseline.save(epochs_baseline_cropped_fname)
# epochs = mne.concatenate_epochs([epochs_A,epochs_B,epochs_X])
# epochs.save(epochs_cropped_fname)




# epochs_A.save(epochs_A_cropped_fname)

#final crop to remove baselines
# epochs_A = mne.read_epochs(epochs_A_fname)
# epochs_B = mne.read_epochs(epochs_B_fname)
# epochs_X = mne.read_epochs(epochs_X_fname)
# epochs_A = epochs_A.crop(-0.2,0.5)
# epochs_B = epochs_B.crop(-0.2,0.5)
# epochs_X = epochs_X.crop(-0.2,0.5)
# epochs_A.baseline=(-0.2,0)
# epochs_B.baseline=(-0.2,0)
# epochs_X.baseline=(-0.2,0)
# epochs_A.save(epochs_A_cropped_fname)
# epochs_B.save(epochs_B_cropped_fname)
# epochs_X.save(epochs_X_cropped_fname)
#
#
# ### does not work unless the baseliens are edited to be the same
# epochs = mne.concatenate_epochs([epochs_A,epochs_B,epochs_X])
# epochs.save(epochs_cropped_fname)
# assert(len(epochs.events) == len(new_trial_info))
#
#
#
# epochs_A = mne.read_epochs(epochs_A_cropped_fname)
# epochs_B = mne.read_epochs(epochs_B_cropped_fname)
# epochs_X = mne.read_epochs(epochs_X_cropped_fname)


# if evoked is made, load
# else make evoked to check for auditory response
# if op.isfile(evoked_fname):
#     evoked = Evoked(evoked_fname)
# else:
#     evoked = epochs.average()
#     evoked.save(evoked_fname)
#     # check for M100 on all trials
#     evoked.plot()
#
# epochs_A.average().plot()
# epochs_B.average().plot()
# epochs_X.average().plot()
