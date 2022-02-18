# MEG decoding cross_val_multiscore script
# adapted for BIPHON by Ben Lang, ben.lang@nyu.edu, and Laura Gwilliams, leg5@nyu.edu

# modules
import mne
import sklearn
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from mne.decoding import cross_val_multiscore
from mne.decoding import SlidingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
# from jr import scorer_spearman # will have to download git repos & import some stuff
from sklearn.metrics import make_scorer, get_scorer
# np.set_printoptions(threshold=sys.maxsize)


# def my_scaler(x):
#     '''
#     Scale btw 0-1.
#     '''
#     x = np.array(x).astype(float)
#     return (x - (np.min(x)) / (np.max(x) - np.min(x)))
#

subjects = ['A0280','A0318','A0392','A0396','A0416','A0417']
# dec_dir = '/Volumes/bin_battuta/biphon/dec'
dec_dir = '/Volumes/hecate/biphon/dec'

# grp_scores = []
# grp_probas = []
# rgs_list = ['ah','ae','i','u','uu','ob','y','yih','vowel_grp1','vowel_grp2','vowel_grp3','nativeness']
# rgs_list = ['ah']
# regressor = 'ah'

# paths
# for regressor in rgs_list:
for subject in subjects:
    # epoch_fname = os.path.join(dec_dir, subject,'%s_biphon-epo.fif'%subject)
    # epochs_baseline_fname = os.path.join(dec_dir,subject,'%s_biphon_baseline_cropped-epo.fif'%subject)

    # load epochs
    epochs_fname = os.path.join(dec_dir, subject,'%s_biphon-epo.fif'%subject)
    epochs = mne.read_epochs(epochs_fname)
    trial_info = epochs.metadata
    if subject[0] == 'A':
        ch = 208

    # extract data
    X = epochs._data[:, 0:ch, :]
    # X = sub_epochs_iy_combo._data[:, 0:ch, :]

    # for regressor in rgs_list:
    y = trial_info[regressor].values
    # y = sub_trial_info_iy_combo['y'].values

    # X = epochs_baseline._data
    # y = epochs_baseline.metadata['nativeness'].values

    # y = my_scaler(y)

    # make classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
    scorer = 'roc_auc'
    score = 'AUC'
    n_jobs=-1
    cv = StratifiedKFold(2)
    slider = SlidingEstimator(n_jobs=n_jobs, scoring=scorer, base_estimator=clf)

    # clf = make_pipeline(StandardScaler(), Ridge())
    # scorer = make_scorer(get_scorer(scorer_spearman))
    # score = 'Spearman R'
    # n_jobs=1
    # cv = StratifiedKFold(5, shuffle=True)
    # slider = SlidingEstimator(n_jobs=n_jobs, scoring=scorer, base_estimator=clf)

    # clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
    # scorer = 'roc_auc'
    # n_jobs=1
    # cv = StratifiedKFold(5, shuffle=True)
    # slider = SlidingEstimator(n_jobs=n_jobs, scoring=scorer, base_estimator=clf)

    # get accuracies
    print ('Fitting %s for %s.'%(regressor,subject))
    scores = cross_val_multiscore(slider, X, y, cv=cv)

    scores = np.mean(scores, axis=0)
    grp_scores.append(scores)

# save raw scores of one regressor for all subjects
scores_arr = np.array(grp_scores)
np.save(dec_dir + '/grp/raw/grp_scores_%s_%s.npy'%(len(subjects),regressor), scores_arr)

# save avg and sem scores for grp
grp_avg = np.mean( np.array(scores_arr), axis=0 )
grp_sem = np.std( np.array(grp_scores), axis=0 ) / np.sqrt(len(grp_scores))
np.save(dec_dir + '/grp/sem/grp_sem_scores_%s_%s.npy'%(len(subjects),regressor), scores_arr)
np.save(dec_dir + '/grp/avg/grp_avg_scores_%s_%s.npy'%(len(subjects),regressor), scores_arr)


### PLOTTING
# Plot average decoding scores of 5 splits
fig, ax = plt.subplots(1)
# ax.plot(epochs.times, grp_probas.mean(0), label='score')
#  ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.axvline(0, color='k')
ax.axvspan(100, 150, alpha=0.5, color='gainsboro')
# ax.set_xlabel('Time (ms)')
# ax.set_ylabel('Probability')
ax.xaxis.set_ticks(np.arange(-200,500,50))
# plt.legend().set_title('%s'%subject)
ax.legend().set_title('All Subj')
# ax.legend(['/y/','/ʏ/'])
# ax.legend(['/ø/','/ɯ/'])
# ax.set_title('Decoding probability of /i/ from /y/ and /ʏ/')
# ax.set_title('Decoding probability of /u/ from /ø/ and /ɯ/')
plt.show()
