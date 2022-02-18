## reject epochs script
# author: Ben Lang, bcl267@nyu.edu

import os
import os.path as op
import mne
import pandas as pd
import numpy as np
from mne import (pick_types, find_events, Epochs, Evoked, compute_covariance,
                 write_cov, read_cov, setup_source_space, write_source_spaces, make_bem_model, make_bem_solution, make_bem_solution, make_forward_solution,
                 write_forward_solution, read_forward_solution, write_bem_solution, convert_forward_solution, read_epochs, grade_to_vertices,
                 compute_source_morph, compute_morph_matrix, read_source_estimate, convert_forward_solution)
# import wx
# from logfile_parse import df as trial_info
# %gui qt

# params
subject = 'A0280'
# meg_dir = '/Volumes/MEG/NYUAD-Lab-Server/DataAnalysis/Ben/MEG/biphon/meg/%s/'%(subject) # change to local meg folder
meg_dir = '/Volumes/bin_battuta/biphon/meg/%s/'%(subject)

# meg_dir = '/scratch/bcl267/DataAnalysis/%s/'%(subject) # change to local meg folder

filt_l = 0
filt_h = 60
tmin = -0.2
tmax = 1.5

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

## reject epochs
trial_info = pd.read_csv(trial_info_fname)
trial_info_A = trial_info[trial_info['position']==1]
trial_info_B = trial_info[trial_info['position']==2]
trial_info_X = trial_info[trial_info['position']==3]


from eelbrain.gui import select_epochs
from eelbrain.load import unpickle

    #### A TRIALS
epochs_A = mne.read_epochs(epochs_A_fname)

if op.isfile(pickled_rej_A_fname):
    rejfile = unpickle(pickled_rej_A_fname)
else:
    select_epochs(epochs_A, vlim=2e-12, mark=['MEG 087','MEG 130'])
    # This reminds you how to name the file and also stops the loop until you press enter
    raw_input('NOTE: Save as subj_rejfile.pickled. \nPress enter when you are done rejecting epochs in the GUI...')
    rejfile = unpickle(pickled_rej_A_fname)

# create a mask to reject the bad epochs
rejs_A = rejfile['accept'].x

# mask epochs and info
epochs_A = epochs_A[rejs_A]
trial_info_A = trial_info_A[rejs_A]

log_mask = np.where(trial_info_A.reject==1, True, False)
epochs_A = epochs_A[log_mask]
trial_info_A = trial_info_A[log_mask]

# set metadata: allows you to specify more complex info about events,
# can use pandas-style queries to access subsets of data

epochs_A.metadata = trial_info_A
epochs_A.save(epochs_A_fname)

# SANITY CHECK!!:
assert(len(epochs_A.events) == len(trial_info_A))
trial_info_A.to_csv(meg_dir+'%s_biphon_A_rej_trialinfo.csv'%(subject))

del epochs_A, rejfile


#### B TRIALS
epochs_B = mne.read_epochs(epochs_B_fname)
if op.isfile(pickled_rej_B_fname):
    rejfile = unpickle(pickled_rej_B_fname)
else:
    select_epochs(epochs_B, vlim=2e-12, mark=['MEG 087','MEG 130'])
    # This reminds you how to name the file and also stops the loop until you press enter
    raw_input('NOTE: Save as subj_rejfile.pickled. \nPress enter when you are done rejecting epochs in the GUI...')
    rejfile = unpickle(pickled_rej_B_fname)

# create a mask to reject the bad epochs
rejs_B = rejfile['accept'].x

# mask epochs and info
epochs_B = epochs_B[rejs_B]
trial_info_B = trial_info_B[rejs_B]

log_mask = np.where(trial_info_B.reject==1, True, False)
epochs_B = epochs_B[log_mask]
trial_info_B = trial_info_B[log_mask]

# set metadata: allows you to specify more complex info about events,
# can use pandas-style queries to access subsets of data

epochs_B.metadata = trial_info_B
epochs_B.save(epochs_B_fname)

# SANITY CHECK!!:
assert(len(epochs_B.events) == len(trial_info_B))
trial_info_B.to_csv(meg_dir+'%s_biphon_B_rej_trialinfo.csv'%(subject))

del epochs_B, rejfile

##### X TRIALS
epochs_X = mne.read_epochs(epochs_X_fname)
if op.isfile(pickled_rej_X_fname):
    rejfile = unpickle(pickled_rej_X_fname)
else:
    select_epochs(epochs_X, vlim=2e-12, mark=['MEG 087','MEG 130'])
    # This reminds you how to name the file and also stops the loop until you press enter
    raw_input('NOTE: Save as subj_rejfile.pickled. \nPress enter when you are done rejecting epochs in the GUI...')
    rejfile = unpickle(pickled_rej_X_fname)

# create a mask to reject the bad epochs
rejs_X = rejfile['accept'].x

# mask epochs and info
epochs_X = epochs_X[rejs_X]
trial_info_X = trial_info_X[rejs_X]

log_mask = np.where(trial_info_X.reject==1, True, False)
epochs_X = epochs_X[log_mask]
trial_info_X = trial_info_X[log_mask]

# set metadata: allows you to specify more complex info about events,
# can use pandas-style queries to access subsets of data

epochs_X.metadata = trial_info_X
epochs_X.save(epochs_X_fname)

# SANITY CHECK!!:
assert(len(epochs_X.events) == len(trial_info_X))
trial_info_X.to_csv(meg_dir+'%s_biphon_X_rej_trialinfo.csv'%(subject))

del epochs_X, rejfile

## make a big csv with all epochs
new_trial_info = pd.concat([trial_info_A,trial_info_B,trial_info_X])
new_trial_info.to_csv(meg_dir+'%s_biphon_rej_trialinfo.csv'%(subject))

# epochs = mne.read_epochs(epochs_fname)
# trial_info = pd.read_csv('/Volumes/bin_battuta/biphon/meg/A0396/A0396_biphon_rej_trialinfo.csv')
# epochs.metadata = trial_info
# epochs = epochs.crop(-0.2,0)
# epochs_A = epochs['vowel_position == 1']
# epochs_A.save(meg_dir + subject + '_biphon_cropped_baseline-epo.fif')
# epochs_A = epochs_A.crop(0,0.5)
# epochs = mne.concatenate_epochs([epochs_A,epochs_B,epochs_X])


#-------------------------------------------------------------------------------
import sys
from mne.io import read_raw_fif
from mne.preprocessing.ica import read_ica
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


# slice out the epochs from A because they are the entire trial period
epochs_A = mne.read_epochs(epochs_A_fname)
# epochs_A = epochs['vowel_position == 1']

# step 7- make noise covariance matrix
if not op.isfile(cov_fname):
    print ('Computing noise covariance for %s.' %(subject)) # number of events
    noise_cov = compute_covariance(epochs_A, tmax=0., method=['shrunk'])
    write_cov(cov_fname, noise_cov)
else:
    noise_cov = read_cov(cov_fname)

#check noise whitening
evoked = epochs_A.average()
evo_whi = evoked.plot_white(noise_cov, time_unit='s')
evo_whi.savefig(evo_whi_fname, dpi=150)
del evoked

# if using native MRI, need to make_bem_model
if not op.isfile(bem_fname):
    print ('Making and writing BEM for %s.' %(subject)) # number of events
    surfaces = make_bem_model(subject, ico=4, conductivity=([0.3]), verbose=True)
    bem = make_bem_solution(surfaces)
    write_bem_solution(bem_fname, bem)

# step 8- make forward solution
if not op.isfile(fwd_fname):
    if not op.isfile(src_fname):
        print ('Making and writing forward solution for %s.' %(subject)) # number of events
        src = setup_source_space(subject, spacing='ico4')
        write_source_spaces(src_fname, src, overwrite=True, verbose=None)
        fwd = make_forward_solution(epochs_A.info, trans_fname, src, bem_fname,
                                 meg=True, ignore_ref=True)
        fwd = convert_forward_solution(fwd,force_fixed=True)
        write_forward_solution(fwd_fname, fwd, overwrite=True, verbose=None)
    else:
        src = src_fname
        fwd = make_forward_solution(epochs_A.info, trans_fname, src, bem_fname,
                                 meg=True, ignore_ref=True)
        fwd = convert_forward_solution(fwd,force_fixed=True)
        write_forward_solution(fwd_fname, fwd, overwrite=True, verbose=None)

else:
    fwd = read_forward_solution(fwd_fname)
    fwd = convert_forward_solution(fwd,force_fixed=True) #missing from previous exps... FUCKING GREAT!!!

# step 9- make inverse solution for epochs
if not op.isfile(inv_fname):
    print ('Making and writing inverse operator for %s.' %(subject)) # number of events
    inverse_operator = make_inverse_operator(epochs_A.info, fwd, noise_cov,
                                         loose=0, depth=None, fixed=True)
    write_inverse_operator(inv_fname, inverse_operator, verbose=None)
else:
    inverse_operator = read_inverse_operator(inv_fname, verbose=None)


#final crop to remove baselines
epochs_A = mne.read_epochs(epochs_A_fname)
epochs_B = mne.read_epochs(epochs_B_fname)
epochs_X = mne.read_epochs(epochs_X_fname)
epochs_A = epochs_A.crop(-0.2,0.5)
epochs_B = epochs_B.crop(-0.2,0.5)
epochs_X = epochs_X.crop(-0.2,0.5)
epochs_A.baseline=(-0.2,0)
epochs_B.baseline=(-0.2,0)
epochs_X.baseline=(-0.2,0)
epochs_A.save(epochs_A_cropped_fname)
epochs_B.save(epochs_B_cropped_fname)
epochs_X.save(epochs_X_cropped_fname)


### does not work unless the baselines are edited to be the same
epochs = mne.concatenate_epochs([epochs_A,epochs_B,epochs_X])
epochs.save(epochs_cropped_fname)
assert(len(epochs.events) == len(new_trial_info))



epochs_A = mne.read_epochs(epochs_A_cropped_fname)
epochs_B = mne.read_epochs(epochs_B_cropped_fname)
epochs_X = mne.read_epochs(epochs_X_cropped_fname)


# if evoked is made, load
# else make evoked to check for auditory response
if op.isfile(evoked_fname):
    evoked = Evoked(evoked_fname)
else:
    evoked = epochs.average()
    evoked.save(evoked_fname)
    # check for M100 on all trials
    evoked.plot()

epochs_A.average().plot()
epochs_B.average().plot()
epochs_X.average().plot()
