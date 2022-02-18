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
    meg_dir = '/Volumes/bin_battuta/biphon/meg/%s/'%(subject)
    os.environ["SUBJECTS_DIR"] = '/Volumes/bin_battuta/biphon/mri/' # change to local mri folder
    epochs_A_fname = meg_dir + subject+ '_biphon_A-epo.fif'
    epochs_B_fname = meg_dir + subject+ '_biphon_B-epo.fif'
    epochs_X_fname = meg_dir + subject+ '_biphon_X-epo.fif'
    epochs_A_cropped_fname = meg_dir + subject + '_biphon_A_cropped-epo.fif'
    epochs_B_cropped_fname = meg_dir + subject + '_biphon_B_cropped-epo.fif'
    epochs_X_cropped_fname = meg_dir + subject + '_biphon_X_cropped-epo.fif'
    evoked_A_fname = meg_dir + subject+ '_biphon_A-evoked-ave.fif'
    evoked_B_fname = meg_dir + subject+ '_biphon_B-evoked-ave.fif'
    evoked_X_fname = meg_dir + subject+ '_biphon_X-evoked-ave.fif'

    # do the correction
    epochs_A = mne.read_epochs(epochs_A_cropped_fname)
    epochs_B = mne.read_epochs(epochs_B_cropped_fname)
    epochs_X = mne.read_epochs(epochs_X_cropped_fname)
    event_id_A = dict(i1_A=10, i2_A=11, u1_A=12, u2_A=13, ah1_A=16, ah2_A=17, ae1_A=14, ae2_A=15, yih1_A=18, yih2_A=19, y1_A=20, y2_A=21, ob1_A=22, ob2_A=23, uu1_A=24, uu2_A=25)
    event_id_B = dict(i1_B=26, i2_B=27, u1_B=28, u2_B=29, ah1_B=30, ah2_B=31, ae1_B=32, ae2_B=33, yih1_B=34, yih2_B=35, y1_B=36, y2_B=37, ob1_B=38, ob2_B=39, uu1_B=40, uu2_B=41)
    event_id_X = dict(i1_X=42, i2_X=43, u1_X=44, u2_X=45, ah1_X=46, ah2_X=47, ae1_X=48, ae2_X=49, yih1_X=50, yih2_X=51, y1_X=52, y2_X=53, ob1_X=54, ob2_X=55, uu1_X=56, uu2_X=57)
    epochs_A.event_id = event_id_A
    epochs_B.event_id = event_id_B
    epochs_X.event_id = event_id_X
    epochs_A.save(epochs_A_cropped_fname)
    epochs_B.save(epochs_B_cropped_fname)
    epochs_X.save(epochs_X_cropped_fname)
