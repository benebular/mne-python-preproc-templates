## Authors: Ben Lang & Julien Dirani
# e: ben.lang@nyu.edu

import numpy as np
import pandas as pd
import os
# import eelbrain
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
        # for plotting
        # meg_dir = '/scratch/bcl267/DataAnalysis/%s/'%(subject) # change to local meg folder
        meg_dir = '/Volumes/bin_battuta/biphon/meg/%s/'%(subject)
        # meg_dir = '/scratch/bcl267/DataAnalysis/%s/'%(subject) # change to local meg folder

        # for plotting
        os.environ["SUBJECTS_DIR"] = '/Volumes/bin_battuta/biphon/mri/' # change to local mri folder

        # file names, MEG preproc
        epochs_A_cropped_fname = meg_dir + subject + '_biphon_A_cropped-epo.fif'
        epochs_B_cropped_fname = meg_dir + subject + '_biphon_B_cropped-epo.fif'
        epochs_X_cropped_fname = meg_dir + subject + '_biphon_X_cropped-epo.fif'
        evoked_A_fname = meg_dir + subject+ '_biphon_A-evoked-ave.fif'
        evoked_B_fname = meg_dir + subject+ '_biphon_B-evoked-ave.fif'
        evoked_X_fname = meg_dir + subject+ '_biphon_X-evoked-ave.fif'

        # filenames, MRI/source space creation
        mri_dir = '/Volumes/bin_battuta/biphon/mri/%s/bem/'%subject # subject's bem folder
        inv_fname = os.path.join(meg_dir, '%s_biphon-inv.fif'%subject)

        # STCs
        evoked_dir = '/Volumes/bin_battuta/biphon/evoked/%s/'%subject
        stc_fname = meg_dir + subject+ '_biphon.stc.npy'
        evoked_stc_fname = meg_dir + subject + '_evoked'
        evoked_native_stc_fname = meg_dir + subject + '_evoked_native'
        evoked_nonnative_stc_fname = meg_dir + subject + '_evoked_nonnative'
        stc_epochs_fname = '/Volumes/bin_battuta/biphon/stc/%s/A0396_biphon_stc_epochs.stc.npy'%subject
        stc_morphed_native_fname = meg_dir + subject + '_morphed_native'
        stc_morphed_nonnative_fname = meg_dir + subject + '_morphed_nonnative'
        stc_morphed_fname = meg_dir + subject + '_morphed_all'

        epochs_A = mne.read_epochs(epochs_A_cropped_fname)
        epochs_B = mne.read_epochs(epochs_B_cropped_fname)
        epochs_X = mne.read_epochs(epochs_X_cropped_fname)

        epochs_A_native = epochs_A['nativeness == 1']
        epochs_A_nonnative = epochs_A['nativeness == 0']
        epochs_B_native = epochs_B['nativeness == 1']
        epochs_B_nonnative = epochs_B['nativeness == 0']
        epochs_X_native = epochs_X['nativeness == 1']
        epochs_X_nonnative = epochs_X['nativeness == 0']

        ## EVOKED
        vowel_labels = ['i1_A','i2_A','u1_A','u2_A', 'ah1_A','ah2_A','ae1_A','ae2_A',
                    'i1_B','i2_B','u1_B','u2_B','ah1_B','ah2_B','ae1_B','ae2_B',
                    'i1_X','i2_X','u1_X','u2_X','ah1_X','ah2_X','ae1_X','ae2_X',
                    'ih1_A','ih2_A','ih1_B','ih2_B','ih1_X','ih2_X',
                    'yih1_A','yih2_A','yih1_B','yih2_B','yih1_X','yih2_X',
                    'y1_A','y2_A','y1_B','y2_B','y1_X','y2_X',
                    'uu1_A','uu2_A','uu1_B','uu2_B','uu1_X','uu2_X',
                    'ob1_A','ob2_A','ob1_B','ob2_B','ob1_X','ob2_X']

        # evoked_all= (epochs_A['i1_A','i2_A','u1_A','u2_A', 'ah1_A','ah2_A','ae1_A','ae2_A',
        #                 'i1_B','i2_B','u1_B','u2_B','ah1_B','ah2_B','ae1_B','ae2_B',
        #                 'i1_X','i2_X','u1_X','u2_X','ah1_X','ah2_X','ae1_X','ae2_X',
        #                 'ih1_A','ih2_A','ih1_B','ih2_B','ih1_X','ih2_X',
        #                 'yih1_A','yih2_A','yih1_B','yih2_B','yih1_X','yih2_X',
        #                 'y1_A','y2_A','y1_B','y2_B','y1_X','y2_X',
        #                 'uu1_A','uu2_A','uu1_B','uu2_B','uu1_X','uu2_X',
        #                 'ob1_A','ob2_A','ob1_B','ob2_B','ob1_X','ob2_X'].average().plot())
        #
        native_groups_A = [['i1_A','i2_A'],['u1_A','u2_A'],['ah1_A','ah2_A'],['ae1_A','ae2_A']]
        native_groups_B = [['i1_B','i2_B'],['u1_B','u2_B'],['ah1_B','ah2_B'],['ae1_B','ae2_B']]
        native_groups_X = [['i1_X','i2_X'],['u1_X','u2_X'],['ah1_X','ah2_X'],['ae1_X','ae2_X']]

        nonnative_groups_A = [['i1_A','i2_A'],['u1_A','u2_A'],['ah1_A','ah2_A'],['ae1_A','ae2_A'],['y1_A','y2_A'],['yih1_A','yih2_A'],['ob1_A','ob2_A'],['uu1_A','uu2_A']]
        nonnative_groups_B = [['i1_B','i2_B'],['u1_B','u2_B'],['ah1_B','ah2_B'],['ae1_B','ae2_B'],['y1_B','y2_B'],['yih1_B','yih2_B'],['ob1_B','ob2_B'],['uu1_B','uu2_B']]
        nonnative_groups_X = [['i1_X','i2_X'],['u1_X','u2_X'],['ah1_X','ah2_X'],['ae1_X','ae2_X'],['y1_X','y2_X'],['yih1_X','yih2_X'],['ob1_X','ob2_X'],['uu1_X','uu2_X']]

        inv_fname = os.path.join(meg_dir, '%s_biphon-inv.fif'%subject)
        inverse_operator = read_inverse_operator(inv_fname, verbose=None)
        snr = 3.0  # Standard assumption for average data but using it for single trial
        lambda2 = 1.0 / snr ** 2
        A_native_commentlist = []
        A_nonnative_commentlist = []
        B_native_commentlist = []
        B_nonnative_commentlist = []
        X_native_commentlist = []
        X_nonnative_commentlist = []


        for g in native_groups_A:
            ev = epochs_A_native[g].average()
            ev.comment = "_".join(g)
            print ('Applying inverse to create evoked STC for %s.' %(g)) # number of events
            stc_evoked = apply_inverse(ev, inverse_operator, lambda2, method='dSPM')
            # stc_evoked.save(evoked_dir + '%s'%(evoked))
            # Morph
            morph = mne.compute_source_morph(stc_evoked, spacing=4)
            print ('Morphing %s.' %(g)) # number of events
            stc_fsaverage = morph.apply(stc_evoked)
            A_native_commentlist.append(ev.comment)
            stc_fsaverage.save(evoked_dir + subject + '_native_' + ev.comment + '_morphed')

        for g in native_groups_B:
            ev = epochs_B_native[g].average()
            ev.comment = "_".join(g)
            print ('Applying inverse to create evoked STC for %s.' %(g)) # number of events
            stc_evoked = apply_inverse(ev, inverse_operator, lambda2, method='dSPM')
            # stc_evoked.save(evoked_dir + '%s'%(evoked))
            # Morph
            morph = mne.compute_source_morph(stc_evoked, spacing=4)
            print ('Morphing %s.' %(g)) # number of events
            stc_fsaverage = morph.apply(stc_evoked)
            B_native_commentlist.append(ev.comment)
            stc_fsaverage.save(evoked_dir + subject + '_native_' + ev.comment + '_morphed')

        for g in native_groups_X:
            ev = epochs_X_native[g].average()
            ev.comment = "_".join(g)
            print ('Applying inverse to create evoked STC for %s.' %(g)) # number of events
            stc_evoked = apply_inverse(ev, inverse_operator, lambda2, method='dSPM')
            # stc_evoked.save(evoked_dir + '%s'%(evoked))
            # Morph
            morph = mne.compute_source_morph(stc_evoked, spacing=4)
            print ('Morphing %s.' %(g)) # number of events
            stc_fsaverage = morph.apply(stc_evoked)
            X_native_commentlist.append(ev.comment)
            stc_fsaverage.save(evoked_dir + subject + '_native_' + ev.comment + '_morphed')

        for g in nonnative_groups_A:
            ev = epochs_A_nonnative[g].average()
            ev.comment = "_".join(g)
            print ('Applying inverse to create evoked STC for %s.' %(g)) # number of events
            stc_evoked = apply_inverse(ev, inverse_operator, lambda2, method='dSPM')
            # stc_evoked.save(evoked_dir + '%s'%(evoked))
            # Morph
            morph = mne.compute_source_morph(stc_evoked, spacing=4)
            print ('Morphing %s.' %(g)) # number of events
            stc_fsaverage = morph.apply(stc_evoked)
            A_nonnative_commentlist.append(ev.comment)
            stc_fsaverage.save(evoked_dir + subject + '_nonnative_' + ev.comment + '_morphed')

        for g in nonnative_groups_B:
            ev = epochs_B_nonnative[g].average()
            ev.comment = "_".join(g)
            print ('Applying inverse to create evoked STC for %s.' %(g)) # number of events
            stc_evoked = apply_inverse(ev, inverse_operator, lambda2, method='dSPM')
            # stc_evoked.save(evoked_dir + '%s'%(evoked))
            # Morph
            morph = mne.compute_source_morph(stc_evoked, spacing=4)
            print ('Morphing %s.' %(g)) # number of events
            stc_fsaverage = morph.apply(stc_evoked)
            B_nonnative_commentlist.append(ev.comment)
            stc_fsaverage.save(evoked_dir + subject + '_nonnative_' + ev.comment + '_morphed')

        for g in nonnative_groups_X:
            ev = epochs_X_nonnative[g].average()
            ev.comment = "_".join(g)
            print ('Applying inverse to create evoked STC for %s.' %(g)) # number of events
            stc_evoked = apply_inverse(ev, inverse_operator, lambda2, method='dSPM')
            # stc_evoked.save(evoked_dir + '%s'%(evoked))
            # Morph
            morph = mne.compute_source_morph(stc_evoked, spacing=4)
            print ('Morphing %s.' %(g)) # number of events
            stc_fsaverage = morph.apply(stc_evoked)
            X_nonnative_commentlist.append(ev.comment)
            stc_fsaverage.save(evoked_dir + subject + '_nonnative_' + ev.comment + '_morphed')




        # stc_morphed = you make your thingy
        # stc_morphed.save(dir + ev.comment + '.stc')


# evoked_native = (epochs['i1_A' ,'i2_A', 'u1_A' ,'u2_A', 'ah1_A' ,'ah2_A' ,'ae1_A', 'ae2_A' ,
#                 'i1_B' ,'i2_B', 'u1_B' ,'u2_B' ,'ah1_B' ,'ah2_B' ,'ae1_B' ,'ae2_B' ,
#                 'i1_X' ,'i2_X', 'u1_X' ,'u2_X', 'ah1_X' ,'ah2_X' ,'ae1_X', 'ae2_X',
#                 'ih1_A' ,'ih2_A' ,'ih1_B' ,'ih2_B' ,'ih1_X' ,'ih2_X'].average().plot())
# evoked_nonnative = (epochs['yih1_A','yih2_A','yih1_B','yih2_B','yih1_X','yih2_X',
#                 'y1_A','y2_A','y1_B','y2_B','y1_X','y2_X',
#                 'uu1_A','uu2_A','uu1_B','uu2_B','uu1_X','uu2_X',
#                 'ob1_A','ob2_A','ob1_B','ob2_B','ob1_X','ob2_X'].average().plot())

# ##Evoked by vowel_position
# epochs_A = epochs['vowel_position == 1']
# epochs_B = epochs['vowel_position == 2']
# epochs_X = epochs['vowel_position == 3']
# evoked_A = epochs_A.average().plot()
# evoked_B = epochs_B.average().plot()
# evoked_X = epochs_X.average().plot()


# # Native
# evoked_native = epochs['native'].average()
# evoked_native.save(evoked_native_fname)
# evoked_native = Evoked('/Volumes/bin_battuta/biphon/meg/A0280/A0280_biphon-evoked-native-ave.fif')
#     # check for M100
# evoked_native.plot()
# plt.xticks(np.arange(0, 2, 0.05))
# plt.title('Evoked Response for Native Trials')
#
#
#
# # Nonnative
# evoked_nonnative = epochs['nonnative'].average()
# evoked_nonnative.save(evoked_nonnative_fname)
# evoked_nonnative = Evoked('/Volumes/bin_battuta/biphon/meg/A0280/A0280_biphon-evoked-nonnative-ave.fif')
#
#     # check for M100
# evoked_nonnative.plot()
# plt.xticks(np.arange(0, 2, 0.05))
# plt.title('Evoked Response for Non-native Trials')


# step 10- make source estimates
# snr = 3.0  # Standard assumption for average data but using it for single trial
# lambda2 = 1.0 / snr ** 2

# apply inverse to epochs
# stc_epochs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method='dSPM')
# np.save(file=stc_epochs_fname, arr=stc_epochs)

# morph = mne.compute_source_morph(stc_epochs, spacing=4, subject_to='fsaverage', subject_from=subject)
# print ('Morphing %s.' %(subject)) # number of events
# stc_epochs_fsaverage = morph.apply(stc_epochs)
# stc_epochs_fsaverage.save(stc_morphed_fname, overwrite=True)



# stc_epochs_A = apply_inverse_epochs(epochs_A, inverse_operator, lambda2, method='dSPM')
# stc_evoked_A = apply_inverse(evoked_A, inverse_operator, lambda2, method='dSPM', pick_ori='normal')
# stc_evoked_A.save(evoked_A_stc_fname)
# plot_source_estimates(stc_evoked_A, subject=subject, surface='inflated', hemi='lh', smoothing_steps=5, time_viewer=True)

# All stcs
# evoked_list = [evoked_i_A_native,
                # evoked_u_A_native,
                # evoked_ah_A_native,
                # evoked_ae_A_native,
                # evoked_i_B_native,
                # evoked_u_B_native,
                # evoked_ah_B_native,
                # evoked_ae_B_native,
                # evoked_i_X_native,
                # evoked_u_X_native,
                # evoked_ah_X_native,
                # evoked_ae_X_native,
                # evoked_i_A_nonnative,
                # evoked_u_A_nonnative,
                # evoked_ah_A_nonnative,
                # evoked_ae_A_nonnative,
                # evoked_y_A_nonnative,
                # evoked_yih_A_nonnative,
                # evoked_ob_A_nonnative,
                # evoked_uu_A_nonnative,
                # evoked_i_B_nonnative,
                # evoked_u_B_nonnative,
                # evoked_ah_B_nonnative,
                # evoked_ae_B_nonnative,
                # evoked_y_B_nonnative,
                # evoked_yih_B_nonnative,
                # evoked_ob_B_nonnative,
                # evoked_uu_B_nonnative,
                # evoked_i_X_nonnative,
                # evoked_u_X_nonnative,
                # evoked_ah_X_nonnative,
                # evoked_ae_X_nonnative,
                # evoked_y_X_nonnative,
                # evoked_yih_X_nonnative,
                # evoked_ob_X_nonnative,
                # evoked_uu_X_nonnative]



# inv_fname = os.path.join(meg_dir, '%s_biphon-inv.fif'%subject)
# inverse_operator = read_inverse_operator(inv_fname, verbose=None)
#
# for evoked in evoked_list:
#     print ('Applying inverse to create evoked STC for %s.' %(evoked)) # number of events
#     stc_evoked = apply_inverse(evoked, inverse_operator, lambda2, method='dSPM')
#     # stc_evoked.save(evoked_dir + '%s'%(evoked))
#
#     # Morph
#     morph = mne.compute_source_morph(stc_evoked, spacing=4)
#     print ('Morphing %s.' %(stc_evoked)) # number of events
#     stc_fsaverage = morph.apply(stc_evoked)
#     stc_fsaverage.save(evoked_dir + '%s_morphed'%(stc_evoked))
#


#apply inverse to evoked, this is not morphed
# # Native stcs
# print ('Applying inverse to create evoked STC for %s.' %(subject)) # number of events
# stc_evoked_native = apply_inverse(evoked_native, inverse_operator, lambda2, method='dSPM', pick_ori='normal')
# stc_evoked_native.save(evoked_native_stc_fname)
#
# # Morph
# morph = mne.compute_source_morph(stc_evoked_native, spacing=4)
# print ('Morphing %s.' %(subject)) # number of events
# stc_fsaverage = morph.apply(stc_evoked_native)
# stc_fsaverage.save(stc_morphed_native_fname)
#
#         # Plot
#         stc_evoked = read_source_estimate('/Volumes/bin_battuta/biphon/meg/A0280/A0280_evoked_native-lh.stc')
#         plot_source_estimates(stc_evoked_native, subject=subject, surface='inflated', hemi='lh', smoothing_steps=5, time_viewer=True)
#
#         stc_morphed_native = read_source_estimate('/Volumes/bin_battuta/biphon/meg/A0280/A0280_morphed_native-lh.stc')
#         plot_source_estimates(stc_morphed_native, subject='fsaverage', surface='inflated', hemi='lh', smoothing_steps=5, time_viewer=True)

# X = morph_stcs(stc_epochs, subject)

# check X is same length as trials
# assert(X.shape[1] == len(trial_info))

# save
# np.save(stc_fname, X)

# import mne
# vertices_to = mne.grade_to_vertices('fsaverage', grade=4, subjects_dir=mri_dir) #fsaverage's source space = target brain
# morph_mat = mne.compute_morph_matrix(subject_from=subject, subject_to='fsaverage', vertices_from=stc_evoked.vertices, vertices_to=vertices_to, subjects_dir=mri_dir)
# stc_morph = mne.morph_data_precomputed(subject_from=subject, subject_to='fsaverage', stc_from=stc_evoked, vertices_to=vertices_to, morph_mat=morph_mat)
# stc_morph.save(stc_epochs_fname + '%s_%s_dSPM' %(subject,idx))


# # plotting
# close the time_viewer window before the brain views to avoid crashing in terminal
# plot_source_estimates(stc_fsaverage, subject=subject, surface='inflated', hemi='lh', smoothing_steps=5)

# if you want to visualise and move along time course, confirm auditory response
# stc_evoked.plot(hemi = 'split', time_viewer=True)
