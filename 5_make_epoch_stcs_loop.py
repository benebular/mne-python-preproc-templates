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
                 compute_source_morph, compute_morph_matrix, read_source_estimate)
# from mne import (pick_types, find_events, Epochs, Evoked, compute_covariance,
#                  write_cov, read_cov, setup_source_space, write_source_spaces, make_bem_model, make_bem_solution, make_bem_solution, make_forward_solution,
#                  write_forward_solution, read_forward_solution, write_bem_solution, convert_forward_solution, read_epochs, grade_to_vertices,
#                  read_source_estimate)
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, apply_inverse, write_inverse_operator, read_inverse_operator
from mne.preprocessing import ICA
from mne.viz import plot_source_estimates
import matplotlib.pyplot as plt
mne.set_log_level(verbose='INFO')
np.set_printoptions(threshold=sys.maxsize)

def make_epoch_stcs(epochs, snr = 2.0, method='dSPM', morph=True, save_to_disk = True):
	"""Apply inverse operator to epochs to get source estimates of each item"""
	lambda2 = 1.0 / snr ** 2.0
	inverse = inverse_operator
	eps = mne.minimum_norm.apply_inverse_epochs(epochs=epochs,inverse_operator=inverse,lambda2=lambda2,method = method)
	if morph == True:
		eps_morphed = []
		counter = 1
		morph_status = 'morphed'
	# create morph map
	# get vertices to morph to (we'll take the fsaverage vertices)
		subject_to = 'fsaverage'
		fs = mne.read_source_spaces(mri_dir + '%s/bem/%s-ico-4-src.fif' %(subject_to, subject_to))
		#vertices_to = [fs[0]['vertno'], fs[1]['vertno']]
		vertices_to = mne.grade_to_vertices('fsaverage', grade=4, subjects_dir=mri_dir)
		subject_from = subject

		for stc_from in eps:
			print ("Morphing source estimate for epoch %d" %counter) #python 3
			# use the morph function
			morph = mne.compute_source_morph(stc_from, subject_from, subject_to, spacing=4, subjects_dir=mri_dir)
			stc = morph.apply(stc_from)
			eps_morphed.append(stc)
			counter += 1
			eps = eps_morphed
	if save_to_disk:
		pass
		#with open(op.join(stc_cont, '%s_stc_epochs.pickled' %subject), 'w') as fileout:
			#pickle.dump(eps, fileout)
	return eps

subjects = ['A0280','A0318','A0392','A0396','A0416','A0417']

## Import data, compute STCs, save individual files

for subject in subjects:
    mri_dir = '/Volumes/bin_battuta/biphon/mri/' # subject's bem folder
    main_exp_dir = '/Volumes/bin_battuta/biphon/stc/%s/'%subject
    meg_dir = '/Volumes/bin_battuta/biphon/meg/%s/'%(subject) # change to local meg folder
    root = '/Volumes/bin_battuta/biphon'
    os.environ["SUBJECTS_DIR"] = '/Volumes/bin_battuta/biphon/mri/' # change to local mri folder

    epochs_cropped_fname = meg_dir + subject + '_biphon_cropped-epo.fif'
    inv_fname = os.path.join(meg_dir, '%s_biphon-inv.fif'%subject)

    epochs = mne.read_epochs(epochs_cropped_fname)
    inverse_operator = read_inverse_operator(inv_fname, verbose=None)

    # trialInfo_fname = os.path.join(root, 'meg/%s/%s_biphon_rej_trialinfo.csv' %(subject,subject))
    # trialInfo = pd.read_csv(trialInfo_fname)
    #
    # assert(len(epochs.events) == len(trialInfo))

    print ('Preparing to morph epoch STCs...')
    eps = make_epoch_stcs(epochs)
    a=0 # starts with zero so that it matches the Pandas df of trialInfo in lmm output script
    # root = '/Volumes/MEG/NYUAD-Lab-Server/DataAnalysis/Ben/MEG/biphon/'
    # for i in trialInfo:
    for ep in eps:
            print ('Saving epoch STCs %s from %s...' %(a, subject))
            ep.save(main_exp_dir + '%s_%s' %(subject,a))
            a=a+1


	# #make evokeds!
	# ev_compound = epochs['compound']
	# ev_reduplicate = epochs['reduplicate']
    #
	# ####make evoked averages!
	# epochs.equalize_event_counts(event_id)
    #
	# ev_compound = epochs['compound'].average()
	# ev_reduplicate = epochs['reduplicate'].average()
    #
	# conditions = [ev_compound, ev_reduplicate]
    #
	# filenames = ['compound', 'reduplicate']
	# i = 0
	# vertices_to = mne.grade_to_vertices('fsaverage', grade=4, mri_dir=mri_dir)
    #
    #
	# for cond in conditions:
    #
	# 	###for epoch stcs
    #
	# 	eps = make_epoch_stcs(cond)
	# 	a = 1
	# 	for ep in eps:
	# 		ep.save(main_exp_dir + 'stc/%s%s%s' %(subject,filenames[i],a))
	# 		a = a+1
	# 	i = i+1
    #
	# 	###for average stcs
    #
	# 	lambda2 = 1.0 / 3.0 ** 2
	# 	my_stc = mne.minimum_norm.apply_inverse(cond,inv,lambda2=lambda2,verbose=False,method='dSPM')
	# 	if i == 0:
	# 		subject_from = subject
	# 		subject_to = 'fsaverage'
	# 		morph = mne.compute_source_morph(my_stc, subject_from, subject_to, mri_dir=mri_dir)
	# 		stc_morphed = morph.apply(my_stc)
	# 		stc_morphed.save(main_exp_dir + 'average_stc_equalized_counts/%s%s' %(subject,filenames[i]))
	# 	else:
	# 		stc_morphed = morph.apply(my_stc)
	# 		stc_morphed.save(main_exp_dir + 'average_stc_equalized_counts/%s%s' %(subject,filenames[i]))
	# 	i = i+1
