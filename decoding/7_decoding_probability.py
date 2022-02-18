# decoding probability script
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
from sklearn.metrics import make_scorer, get_scorer
# np.set_printoptions(threshold=sys.maxsize)

subjects = ['A0280','A0318','A0392','A0396','A0416','A0417']
# subject = 'A0280'

# initialize list for group scores on each vowel
grp_y_proba_scores = []
grp_y_proba_scores2 = []
grp_y_proba_scores3 = []
grp_y_proba_scores4 = []
grp_y_proba_scores5 = []
grp_y_proba_scores6 = []
grp_y_proba_scores7 = []
grp_y_proba_scores8 = []
grp_y_proba_scores9 = []
grp_y_proba_scores10 = []





vowel_cat_list = ['y_yih','ob_uu','i_i','u_u','y_y','yih_yih','ob_ob','uu_uu']

# paths
# dec_dir = '/Volumes/bin_battuta/biphon/dec'
dec_dir = '/Volumes/hecate/biphon/dec'


for subject in subjects:
    # load epochs
    epochs_fname = os.path.join(dec_dir, subject,'%s_biphon-epo.fif'%subject)
    epochs = mne.read_epochs(epochs_fname)
    # trial_info = epochs.metadata
    if subject[0] == 'A':
        ch = 208

    # make four native vowel indexes for the four diametric opposite pairs /i, ah, ae, u/
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #         print(epochs.metadata)

    # 1: i, 0:ah
    epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'i') & ((epochs.metadata['pos1_nospk'] == 'ah') | (epochs.metadata['pos2_nospk'] == 'ah')), 'i_ah_AB'] = 1
    epochs.metadata['i_ah_AB'].fillna(0, inplace=True)
    epochs.metadata['i_ah_AB'] = epochs.metadata['i_ah_AB'].astype(int)
    trials_i_ah_AB = pd.concat([epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'i') & ((epochs.metadata['pos1_nospk'] == 'ah') | (epochs.metadata['pos2_nospk'] == 'ah'))],
        epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'ah') & ((epochs.metadata['pos1_nospk'] == 'i') | (epochs.metadata['pos2_nospk'] == 'i'))]], axis=0)
    idx_i_ah_AB = epochs.metadata.index.isin(trials_i_ah_AB.index)

    # 1: ah, 0: i
    # this is identical to the above, just flipping the 1 and 0
    epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'ah') & ((epochs.metadata['pos1_nospk'] == 'i') | (epochs.metadata['pos2_nospk'] == 'i')), 'ah_i_AB'] = 1
    epochs.metadata['ah_i_AB'].fillna(0, inplace=True)
    epochs.metadata['ah_i_AB'] = epochs.metadata['ah_i_AB'].astype(int)
    # trials_ah_i_AB = pd.concat([epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'ah') & ((epochs.metadata['pos1_nospk'] == 'i') | (epochs.metadata['pos2_nospk'] == 'i'))],
    #     epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'i') & ((epochs.metadata['pos1_nospk'] == 'ah') | (epochs.metadata['pos2_nospk'] == 'ah'))]], axis=0)
    # idx_ah_i_AB = epochs.metadata.index.isin(trials_ah_i_AB.index)

    # 1: u, 0: ae
    epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'u') & ((epochs.metadata['pos1_nospk'] == 'ae') | (epochs.metadata['pos2_nospk'] == 'ae')), 'u_ae_AB'] = 1
    epochs.metadata['u_ae_AB'].fillna(0, inplace=True)
    epochs.metadata['u_ae_AB'] = epochs.metadata['u_ae_AB'].astype(int)
    trials_u_ae_AB = pd.concat([epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'u') & ((epochs.metadata['pos1_nospk'] == 'ae') | (epochs.metadata['pos2_nospk'] == 'ae'))],
        epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'ae') & ((epochs.metadata['pos1_nospk'] == 'u') | (epochs.metadata['pos2_nospk'] == 'u'))]], axis=0)
    idx_u_ae_AB = epochs.metadata.index.isin(trials_u_ae_AB.index)

    # 1: ae, 0: u
    # identical to the above
    epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'ae') & ((epochs.metadata['pos1_nospk'] == 'u') | (epochs.metadata['pos2_nospk'] == 'u')), 'ae_u_AB'] = 1
    epochs.metadata['ae_u_AB'].fillna(0, inplace=True)
    epochs.metadata['ae_u_AB'] = epochs.metadata['ae_u_AB'].astype(int)
    # trials_ae_u_AB = pd.concat([epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'ae') & ((epochs.metadata['pos1_nospk'] == 'u') | (epochs.metadata['pos2_nospk'] == 'u'))],
    #     epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'u') & ((epochs.metadata['pos1_nospk'] == 'ae') | (epochs.metadata['pos2_nospk'] == 'ae'))]], axis=0)
    # idx_ae_u_AB = epochs.metadata.index.isin(trials_ae_u_AB.index)

    # create index against all other vowels
    # 1: i, 0: ah, u, ae
    # epochs.metadata.loc[((epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'i') & (epochs.metadata['nativeness'] == 1)), 'i_all_native_AB'] = 1
    # epochs.metadata['i_all_native_AB'].fillna(0, inplace=True)
    # epochs.metadata['i_all_native_AB'] = epochs.metadata['i_all_native_AB'].astype(int)
    # trials_i_all_native_AB = epochs.metadata.loc[((epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'i') & (epochs.metadata['nativeness'] == 1))]
    # idx_i_all_AB = epochs.metadata.index.isin(trials_i_all_native_AB.index)

    # 1: y and 0: i
    epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'y') & ((epochs.metadata['pos1_nospk'] == 'i') | (epochs.metadata['pos2_nospk'] == 'i')), 'y_i_AB'] = 1
    epochs.metadata['y_i_AB'].fillna(0, inplace=True)
    epochs.metadata['y_i_AB'] = epochs.metadata['y_i_AB'].astype(int)
    trials_y_i_AB = pd.concat([epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'y') & ((epochs.metadata['pos1_nospk'] == 'i') | (epochs.metadata['pos2_nospk'] == 'i'))],
        epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'i') & ((epochs.metadata['pos1_nospk'] == 'y') | (epochs.metadata['pos2_nospk'] == 'y'))]])
    idx_y_i_AB = epochs.metadata.index.isin(trials_y_i_AB.index)

    #1: yih and 0:1
    epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'yih') & ((epochs.metadata['pos1_nospk'] == 'i') | (epochs.metadata['pos2_nospk'] == 'i')), 'yih_i_AB'] = 1
    epochs.metadata['yih_i_AB'].fillna(0, inplace=True)
    epochs.metadata['yih_i_AB'] = epochs.metadata['yih_i_AB'].astype(int)
    trials_yih_i_AB = pd.concat([epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'yih') & ((epochs.metadata['pos1_nospk'] == 'i') | (epochs.metadata['pos2_nospk'] == 'i'))],
        epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'i') & ((epochs.metadata['pos1_nospk'] == 'yih') | (epochs.metadata['pos2_nospk'] == 'yih'))]])
    idx_yih_i_AB = epochs.metadata.index.isin(trials_yih_i_AB.index)

    # 1: ob and 0: u
    epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'ob') & ((epochs.metadata['pos1_nospk'] == 'u') | (epochs.metadata['pos2_nospk'] == 'u')), 'ob_u_AB'] = 1
    epochs.metadata['ob_u_AB'].fillna(0, inplace=True)
    epochs.metadata['ob_u_AB'] = epochs.metadata['ob_u_AB'].astype(int)
    trials_ob_u_AB = pd.concat([epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'ob') & ((epochs.metadata['pos1_nospk'] == 'u') | (epochs.metadata['pos2_nospk'] == 'u'))],
        epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'u') & ((epochs.metadata['pos1_nospk'] == 'ob') | (epochs.metadata['pos2_nospk'] == 'ob'))]])
    idx_ob_u_AB = epochs.metadata.index.isin(trials_ob_u_AB.index)

    #1: uu and 0: u
    epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'uu') & ((epochs.metadata['pos1_nospk'] == 'u') | (epochs.metadata['pos2_nospk'] == 'u')), 'uu_u_AB'] = 1
    epochs.metadata['uu_u_AB'].fillna(0, inplace=True)
    epochs.metadata['uu_u_AB'] = epochs.metadata['uu_u_AB'].astype(int)
    trials_uu_u_AB = pd.concat([epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'uu') & ((epochs.metadata['pos1_nospk'] == 'u') | (epochs.metadata['pos2_nospk'] == 'u'))],
        epochs.metadata.loc[(epochs.metadata['position'] != 3) & (epochs.metadata['vowel_iso'] == 'u') & ((epochs.metadata['pos1_nospk'] == 'uu') | (epochs.metadata['pos2_nospk'] == 'uu'))]])
    idx_uu_u_AB = epochs.metadata.index.isin(trials_uu_u_AB.index)




    # get individual non-native vowels from X position as "ambiguous data" for predictor, create index
    trials_i_y_X = epochs.metadata.loc[(epochs.metadata['position'] == 3) & (epochs.metadata['vowel_iso'] == 'y') & ((epochs.metadata['pos1_nospk'] == 'i') | (epochs.metadata['pos2_nospk'] == 'i'))]
    idx_y = epochs.metadata.index.isin(trials_i_y_X.index)

    trials_i_yih_X = epochs.metadata.loc[(epochs.metadata['position'] == 3) & (epochs.metadata['vowel_iso'] == 'yih') & ((epochs.metadata['pos1_nospk'] == 'i') | (epochs.metadata['pos2_nospk'] == 'i'))]
    idx_yih = epochs.metadata.index.isin(trials_i_yih_X.index)

    trials_u_ob_X = epochs.metadata.loc[(epochs.metadata['position'] == 3) & (epochs.metadata['vowel_iso'] == 'ob') & ((epochs.metadata['pos1_nospk'] == 'u') | (epochs.metadata['pos2_nospk'] == 'u'))]
    idx_ob = epochs.metadata.index.isin(trials_u_ob_X.index)

    trials_u_uu_X = epochs.metadata.loc[(epochs.metadata['position'] == 3) & (epochs.metadata['vowel_iso'] == 'u') & ((epochs.metadata['pos1_nospk'] == 'u') | (epochs.metadata['pos2_nospk'] == 'u'))]
    idx_uu = epochs.metadata.index.isin(trials_u_uu_X.index)

    trials_i_i_X = epochs.metadata.loc[(epochs.metadata['position'] == 3) & (epochs.metadata['vowel_iso'] == 'i') & ((epochs.metadata['pos1_nospk'] == 'ah') | (epochs.metadata['pos2_nospk'] == 'ah'))]
    idx_i = epochs.metadata.index.isin(trials_i_i_X.index)

    trials_u_u_X = epochs.metadata.loc[(epochs.metadata['position'] == 3) & (epochs.metadata['vowel_iso'] == 'u') & ((epochs.metadata['pos1_nospk'] == 'ae') | (epochs.metadata['pos2_nospk'] == 'ae'))]
    idx_u = epochs.metadata.index.isin(trials_u_u_X.index)


    # extract data
    X = epochs._data[:, 0:ch, :]
    # X = sub_epochs_iy_combo._data[:, 0:ch, :]

    # for regressor in rgs_list:
    # y = trial_info[regressor].values
    # y = sub_trial_info_iy_combo['y'].values

    # X = epochs_baseline._data
    # y = epochs_baseline.metadata['nativeness'].values

    # y = my_scaler(y)

    # make classifier
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
    # scorer = 'roc_auc'
    # score = 'AUC'
    # n_jobs=-1
    # cv = StratifiedKFold(2)
    # slider = SlidingEstimator(n_jobs=n_jobs, scoring=scorer, base_estimator=clf)

    n_trial, n_sns, n_times = X.shape
    #  X_reshaped = X.reshape((n_trial, n_sns*n_times))
    # X2_reshaped = X2.reshape((n_trial, n_sns*n_times))
    for vowel_cat in vowel_cat_list:
        ### ---------- y and yih ----------- ###
        if vowel_cat == 'y_yih':
            y = epochs.metadata['i_ah_AB'].values # this assigns a 1 or a 0 based on if the trial was an ABX trial with i or ah in it in positions A or B
            print('Fitting classifier and predicting probability of %s for %s...' %(trials_i_y_X['vowel_iso'].values[0],subject))
            y_proba = np.zeros([len(np.where(idx_y)[0]), n_times])
            #  y_pred = np.zeros([len(np.where(idx_i)[0]), n_times])
            for tt in range(n_times):
            	clf.fit(X[:, 0:ch, tt][idx_i_ah_AB], y[idx_i_ah_AB]) # this fits a classifier of the y variable with values 1 or 0 and essentially assigns the neural data via the same index so now all teh 1s are the neural data fo i and the 0s are the neural data for ah
                # y_pred[:, tt] = clf.predict(X[:, 0:ch, tt][idx_i])[0] # need something here to double check that the native category is being predicted accurately, then move below to see if the non-native category is probable?
            	y_proba[:, tt] = clf.predict_proba(X[:, 0:ch, tt][idx_y])[:,0] # last argument takes every row, first column to give you one proba value, get rid of zero to get both proba values
                # this last predict proba is now taking the neural data for instances wher ethe vowel y was in a non-native trial with i and seeing how well we can predict the vowel y from the general i category from the native data

            print('Averaging %s probability scores...'%subject)
            y_proba_scores = np.mean(y_proba, axis=0)
            grp_y_proba_scores.append(y_proba_scores)


            print('Fitting classifier and predicting probability of %s for %s...' %(trials_i_yih_X['vowel_iso'].values[0],subject))
            y_proba2 = np.zeros([len(np.where(idx_yih)[0]), n_times])
            #  y_pred = np.zeros([len(np.where(idx_i)[0]), n_times])
            for tt in range(n_times):
            	clf.fit(X[:, 0:ch, tt][idx_i_ah_AB], y[idx_i_ah_AB])
                # y_pred[:, tt] = clf.predict(X[:, 0:ch, tt][idx_i])[0] # need something here to double check that the native category is being predicted accurately, then move below to see if the non-native category is probable?
            	y_proba2[:, tt] = clf.predict_proba(X[:, 0:ch, tt][idx_yih])[:,0] # last argument takes every row, first column to give you one proba value, get rid of zero to get both proba values

            print('Averaging %s probability scores...'%subject)
            y_proba_scores2 = np.mean(y_proba2, axis=0)
            grp_y_proba_scores2.append(y_proba_scores2)

        if vowel_cat == 'y_y':
            y = epochs.metadata['y_i_AB'].values
            print('Fitting classifier and predicting probability of baseline %s for %s...' %(trials_i_y_X['vowel_iso'].values[0],subject))
            y_proba7 = np.zeros([len(np.where(idx_y)[0]), n_times])
            #  y_pred = np.zeros([len(np.where(idx_i)[0]), n_times])
            for tt in range(n_times):
            	clf.fit(X[:, 0:ch, tt][idx_y_i_AB], y[idx_y_i_AB])
                # y_pred[:, tt] = clf.predict(X[:, 0:ch, tt][idx_i])[0] # need something here to double check that the native category is being predicted accurately, then move below to see if the non-native category is probable?
            	y_proba7[:, tt] = clf.predict_proba(X[:, 0:ch, tt][idx_y])[:,0] # last argument takes every row, first column to give you one proba value, get rid of zero to get both proba values

            print('Averaging %s probability scores...'%subject)
            y_proba_scores7 = np.mean(y_proba7, axis=0)
            grp_y_proba_scores7.append(y_proba_scores7)

        if vowel_cat == 'yih_yih':
            y = epochs.metadata['yih_i_AB'].values
            print('Fitting classifier and predicting probability of baseline %s for %s...' %(trials_i_yih_X['vowel_iso'].values[0],subject))
            y_proba8 = np.zeros([len(np.where(idx_yih)[0]), n_times])
            #  y_pred = np.zeros([len(np.where(idx_i)[0]), n_times])
            for tt in range(n_times):
            	clf.fit(X[:, 0:ch, tt][idx_yih_i_AB], y[idx_yih_i_AB])
                # y_pred[:, tt] = clf.predict(X[:, 0:ch, tt][idx_i])[0] # need something here to double check that the native category is being predicted accurately, then move below to see if the non-native category is probable?
            	y_proba8[:, tt] = clf.predict_proba(X[:, 0:ch, tt][idx_yih])[:,0] # last argument takes every row, first column to give you one proba value, get rid of zero to get both proba values

            print('Averaging %s probability scores...'%subject)
            y_proba_scores8 = np.mean(y_proba8, axis=0)
            grp_y_proba_scores8.append(y_proba_scores8)


        ### ---------- ob and uu ----------- ###
        if vowel_cat == 'ob_uu':
            y = epochs.metadata['u_ae_AB'].values
            print('Fitting classifier and predicting probability of %s for %s...' %(trials_u_ob_X['vowel_iso'].values[0],subject))
            y_proba3 = np.zeros([len(np.where(idx_ob)[0]), n_times])
            #  y_pred = np.zeros([len(np.where(idx_i)[0]), n_times])
            for tt in range(n_times):
            	clf.fit(X[:, 0:ch, tt][idx_u_ae_AB], y[idx_u_ae_AB])
                # y_pred[:, tt] = clf.predict(X[:, 0:ch, tt][idx_i])[0] # need something here to double check that the native category is being predicted accurately, then move below to see if the non-native category is probable?
            	y_proba3[:, tt] = clf.predict_proba(X[:, 0:ch, tt][idx_ob])[:,0] # last argument takes every row, first column to give you one proba value, get rid of zero to get both proba values

            print('Averaging %s probability scores...'%subject)
            y_proba_scores3 = np.mean(y_proba3, axis=0)
            grp_y_proba_scores3.append(y_proba_scores3)

            print('Fitting classifier and predicting probability of %s for %s...' %(trials_u_uu_X['vowel_iso'].values[0],subject))
            y_proba4 = np.zeros([len(np.where(idx_uu)[0]), n_times])
            #  y_pred = np.zeros([len(np.where(idx_i)[0]), n_times])
            for tt in range(n_times):
            	clf.fit(X[:, 0:ch, tt][idx_u_ae_AB], y[idx_u_ae_AB])
                # y_pred[:, tt] = clf.predict(X[:, 0:ch, tt][idx_i])[0] # need something here to double check that the native category is being predicted accurately, then move below to see if the non-native category is probable?
            	y_proba4[:, tt] = clf.predict_proba(X[:, 0:ch, tt][idx_uu])[:,0] # last argument takes every row, first column to give you one proba value, get rid of zero to get both proba values

            print('Averaging %s probability scores...'%subject)
            y_proba_scores4 = np.mean(y_proba4, axis=0)
            grp_y_proba_scores4.append(y_proba_scores4)

        if vowel_cat == 'ob_ob':
            y = epochs.metadata['ob_u_AB'].values
            print('Fitting classifier and predicting probability of baseline %s for %s...' %(trials_u_ob_X['vowel_iso'].values[0],subject))
            y_proba9 = np.zeros([len(np.where(idx_ob)[0]), n_times])
            #  y_pred = np.zeros([len(np.where(idx_i)[0]), n_times])
            for tt in range(n_times):
            	clf.fit(X[:, 0:ch, tt][idx_ob_u_AB], y[idx_ob_u_AB])
                # y_pred[:, tt] = clf.predict(X[:, 0:ch, tt][idx_i])[0] # need something here to double check that the native category is being predicted accurately, then move below to see if the non-native category is probable?
            	y_proba9[:, tt] = clf.predict_proba(X[:, 0:ch, tt][idx_ob])[:,0] # last argument takes every row, first column to give you one proba value, get rid of zero to get both proba values

            print('Averaging %s probability scores...'%subject)
            y_proba_scores9 = np.mean(y_proba9, axis=0)
            grp_y_proba_scores9.append(y_proba_scores9)

        if vowel_cat == 'uu_uu':
            y = epochs.metadata['uu_u_AB'].values
            print('Fitting classifier and predicting probability of baseline %s for %s...' %(trials_u_uu_X['vowel_iso'].values[0],subject))
            y_proba10 = np.zeros([len(np.where(idx_uu)[0]), n_times])
            #  y_pred = np.zeros([len(np.where(idx_i)[0]), n_times])
            for tt in range(n_times):
            	clf.fit(X[:, 0:ch, tt][idx_uu_u_AB], y[idx_uu_u_AB])
                # y_pred[:, tt] = clf.predict(X[:, 0:ch, tt][idx_i])[0] # need something here to double check that the native category is being predicted accurately, then move below to see if the non-native category is probable?
            	y_proba10[:, tt] = clf.predict_proba(X[:, 0:ch, tt][idx_uu])[:,0] # last argument takes every row, first column to give you one proba value, get rid of zero to get both proba values

            print('Averaging %s probability scores...'%subject)
            y_proba_scores10 = np.mean(y_proba10, axis=0)
            grp_y_proba_scores10.append(y_proba_scores10)


        ## i and u against themselves ###
        if vowel_cat == 'i_i':
            y = epochs.metadata['i_ah_AB'].values
            print('Fitting classifier and predicting probability of baseline %s for %s...' %(trials_i_i_X['vowel_iso'].values[0],subject))
            y_proba5 = np.zeros([len(np.where(idx_i)[0]), n_times])
            #  y_pred = np.zeros([len(np.where(idx_i)[0]), n_times])
            for tt in range(n_times):
            	clf.fit(X[:, 0:ch, tt][idx_i_ah_AB], y[idx_i_ah_AB])
                # y_pred[:, tt] = clf.predict(X[:, 0:ch, tt][idx_i])[0] # need something here to double check that the native category is being predicted accurately, then move below to see if the non-native category is probable?
            	y_proba5[:, tt] = clf.predict_proba(X[:, 0:ch, tt][idx_i])[:,0] # last argument takes every row, first column to give you one proba value, get rid of zero to get both proba values

            print('Averaging %s probability scores...'%subject)
            y_proba_scores5 = np.mean(y_proba5, axis=0)
            grp_y_proba_scores5.append(y_proba_scores5)


        if vowel_cat == 'u_u':
            y = epochs.metadata['u_ae_AB'].values
            print('Fitting classifier and predicting probability of baseline %s for %s...' %(trials_u_u_X['vowel_iso'].values[0],subject))
            y_proba6 = np.zeros([len(np.where(idx_u)[0]), n_times])
            #  y_pred = np.zeros([len(np.where(idx_i)[0]), n_times])
            for tt in range(n_times):
            	clf.fit(X[:, 0:ch, tt][idx_u_ae_AB], y[idx_u_ae_AB])
                # y_pred[:, tt] = clf.predict(X[:, 0:ch, tt][idx_i])[0] # need something here to double check that the native category is being predicted accurately, then move below to see if the non-native category is probable?
            	y_proba6[:, tt] = clf.predict_proba(X[:, 0:ch, tt][idx_u])[:,0] # last argument takes every row, first column to give you one proba value, get rid of zero to get both proba values

            print('Averaging %s probability scores...'%subject)
            y_proba_scores6 = np.mean(y_proba6, axis=0)
            grp_y_proba_scores6.append(y_proba_scores6)

    print('Sanity Check: There are %s entries appended to group.'%(len(grp_y_proba_scores)))


print ('Averaging group scores and calculating SEM...')
grp_sem = np.std( np.array(grp_y_proba_scores), axis=0 ) / np.sqrt(len(grp_y_proba_scores))
grp_avg = np.mean( np.array(grp_y_proba_scores), axis=0 )
grp_sem2 = np.std( np.array(grp_y_proba_scores2), axis=0 ) / np.sqrt(len(grp_y_proba_scores2))
grp_avg2 = np.mean( np.array(grp_y_proba_scores2), axis=0 )
grp_sem3 = np.std( np.array(grp_y_proba_scores3), axis=0 ) / np.sqrt(len(grp_y_proba_scores3))
grp_avg3 = np.mean( np.array(grp_y_proba_scores3), axis=0 )
grp_sem4 = np.std( np.array(grp_y_proba_scores4), axis=0 ) / np.sqrt(len(grp_y_proba_scores4))
grp_avg4 = np.mean( np.array(grp_y_proba_scores4), axis=0 )
grp_sem5 = np.std( np.array(grp_y_proba_scores5), axis=0 ) / np.sqrt(len(grp_y_proba_scores5))
grp_avg5 = np.mean( np.array(grp_y_proba_scores5), axis=0 )
grp_sem6 = np.std( np.array(grp_y_proba_scores6), axis=0 ) / np.sqrt(len(grp_y_proba_scores6))
grp_avg6 = np.mean( np.array(grp_y_proba_scores6), axis=0 )
grp_sem7 = np.std( np.array(grp_y_proba_scores7), axis=0 ) / np.sqrt(len(grp_y_proba_scores7))
grp_avg7 = np.mean( np.array(grp_y_proba_scores7), axis=0 )
grp_sem8 = np.std( np.array(grp_y_proba_scores8), axis=0 ) / np.sqrt(len(grp_y_proba_scores8))
grp_avg8 = np.mean( np.array(grp_y_proba_scores8), axis=0 )
grp_sem9 = np.std( np.array(grp_y_proba_scores9), axis=0 ) / np.sqrt(len(grp_y_proba_scores9))
grp_avg9 = np.mean( np.array(grp_y_proba_scores9), axis=0 )
grp_sem10 = np.std( np.array(grp_y_proba_scores10), axis=0 ) / np.sqrt(len(grp_y_proba_scores10))
grp_avg10 = np.mean( np.array(grp_y_proba_scores10), axis=0 )



print('Saving scores...')
np.save(dec_dir + '/grp/raw/grp_raw_scores_i_proba_%s_%s.npy'%(len(subjects),trials_i_y_X['vowel_iso'].values[0]), y_proba)
np.save(dec_dir + '/grp/avg/grp_avg_scores_i_proba_%s_%s.npy'%(len(subjects),trials_i_y_X['vowel_iso'].values[0]), grp_avg)
np.save(dec_dir + '/grp/sem/grp_sem_scores_i_proba_%s_%s.npy'%(len(subjects),trials_i_y_X['vowel_iso'].values[0]), grp_sem)
np.save(dec_dir + '/grp/raw/grp_raw_scores_i_proba_%s_%s.npy'%(len(subjects),trials_i_yih_X['vowel_iso'].values[0]), y_proba2)
np.save(dec_dir + '/grp/avg/grp_avg_scores_i_proba_%s_%s.npy'%(len(subjects),trials_i_yih_X['vowel_iso'].values[0]), grp_avg2)
np.save(dec_dir + '/grp/sem/grp_sem_scores_i_proba_%s_%s.npy'%(len(subjects),trials_i_yih_X['vowel_iso'].values[0]), grp_sem2)
np.save(dec_dir + '/grp/raw/grp_raw_scores_u_proba_%s_%s.npy'%(len(subjects),trials_u_ob_X['vowel_iso'].values[0]), y_proba3)
np.save(dec_dir + '/grp/avg/grp_avg_scores_u_proba_%s_%s.npy'%(len(subjects),trials_u_ob_X['vowel_iso'].values[0]), grp_avg3)
np.save(dec_dir + '/grp/sem/grp_sem_scores_u_proba_%s_%s.npy'%(len(subjects),trials_u_ob_X['vowel_iso'].values[0]), grp_sem3)
np.save(dec_dir + '/grp/raw/grp_raw_scores_u_proba_%s_%s.npy'%(len(subjects),trials_u_uu_X['vowel_iso'].values[0]), y_proba4)
np.save(dec_dir + '/grp/avg/grp_avg_scores_u_proba_%s_%s.npy'%(len(subjects),trials_u_uu_X['vowel_iso'].values[0]), grp_avg4)
np.save(dec_dir + '/grp/sem/grp_sem_scores_u_proba_%s_%s.npy'%(len(subjects),trials_u_uu_X['vowel_iso'].values[0]), grp_sem4)
np.save(dec_dir + '/grp/raw/grp_raw_scores_i_proba_%s_%s.npy'%(len(subjects),trials_i_i_X['vowel_iso'].values[0]), y_proba5)
np.save(dec_dir + '/grp/avg/grp_avg_scores_i_proba_%s_%s.npy'%(len(subjects),trials_i_i_X['vowel_iso'].values[0]), grp_avg5)
np.save(dec_dir + '/grp/sem/grp_sem_scores_i_proba_%s_%s.npy'%(len(subjects),trials_i_i_X['vowel_iso'].values[0]), grp_sem5)
np.save(dec_dir + '/grp/raw/grp_raw_scores_u_proba_%s_%s.npy'%(len(subjects),trials_u_u_X['vowel_iso'].values[0]), y_proba6)
np.save(dec_dir + '/grp/avg/grp_avg_scores_u_proba_%s_%s.npy'%(len(subjects),trials_u_u_X['vowel_iso'].values[0]), grp_avg6)
np.save(dec_dir + '/grp/sem/grp_sem_scores_u_proba_%s_%s.npy'%(len(subjects),trials_u_u_X['vowel_iso'].values[0]), grp_sem6)
np.save(dec_dir + '/grp/raw/grp_raw_scores_ob_proba_%s_%s.npy'%(len(subjects),trials_u_ob_X['vowel_iso'].values[0]), y_proba9)
np.save(dec_dir + '/grp/avg/grp_avg_scores_ob_proba_%s_%s.npy'%(len(subjects),trials_u_ob_X['vowel_iso'].values[0]), grp_avg9)
np.save(dec_dir + '/grp/sem/grp_sem_scores_ob_proba_%s_%s.npy'%(len(subjects),trials_u_ob_X['vowel_iso'].values[0]), grp_sem9)
np.save(dec_dir + '/grp/raw/grp_raw_scores_uu_proba_%s_%s.npy'%(len(subjects),trials_u_uu_X['vowel_iso'].values[0]), y_proba10)
np.save(dec_dir + '/grp/avg/grp_avg_scores_uu_proba_%s_%s.npy'%(len(subjects),trials_u_uu_X['vowel_iso'].values[0]), grp_avg10)
np.save(dec_dir + '/grp/sem/grp_sem_scores_uu_proba_%s_%s.npy'%(len(subjects),trials_u_uu_X['vowel_iso'].values[0]), grp_sem10)


print("Classification complete.")

### PLOTTING

vowel_cat = 'ob_uu'

for vowel_cat in vowel_cat_list:
    if vowel_cat == 'y_yih':
        # Plot average probability scores for two non-native vowels against one native vowel
        fig, ax = plt.subplots(1, figsize=[20,6])
        ax.plot(epochs.times*1000, grp_avg, label='y prediction', color = '#6699CC')
        ax.fill_between(epochs.times*1000, grp_avg-grp_sem, grp_avg+grp_sem, linewidth=0, color='#6699CC', alpha=0.3)
        ax.plot(epochs.times*1000, grp_avg2, label='yih prediction', color = '#CC6677')
        ax.fill_between(epochs.times*1000, grp_avg2-grp_sem2, grp_avg2+grp_sem2, linewidth=0, color='#CC6677', alpha=0.3)
        ax.plot(epochs.times*1000, grp_avg7, label='y baseline', color = '#fecc5c')
        ax.fill_between(epochs.times*1000, grp_avg7-grp_sem7, grp_avg7+grp_sem7, linewidth=0, color='#fecc5c', alpha=0.3)
        ax.plot(epochs.times*1000, grp_avg8, label='yih baseline', color = '#117733')
        ax.fill_between(epochs.times*1000, grp_avg8-grp_sem8, grp_avg8+grp_sem8, linewidth=0, color='#117733', alpha=0.3)
        ax.axvline(0, color='k')
        ax.axvspan(100, 150, alpha=0.5, color='gainsboro')
        ax.set_xlabel('Time (ms)', fontsize=18)
        ax.set_ylabel('Probability', fontsize=18)
        ax.xaxis.set_ticks(np.arange(-200,500,50))
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.legend(['/y/ prediction','/ʏ/ prediction','/y/ baseline', '/ʏ/ baseline'], prop = {'size':18}, loc='upper right')
        ax.set_title('Decoding probability of /i/ from /y/ and /ʏ/', fontsize=20)
        #plt.savefig(dec_dir + '/grp/plots/grp_i_proba_y_yih.png')
        plt.show()

    else:
        if vowel_cat == 'ob_uu':
            fig, ax = plt.subplots(1, figsize=[20,6])
            ax.plot(epochs.times*1000, grp_avg3, label='ob prediction', color = '#332288')
            ax.fill_between(epochs.times*1000, grp_avg3-grp_sem3, grp_avg3+grp_sem3, linewidth=0, color='#332288', alpha=0.3)
            ax.plot(epochs.times*1000, grp_avg4, label='uu prediction', color = '#AA4499')
            ax.fill_between(epochs.times*1000, grp_avg4-grp_sem4, grp_avg4+grp_sem4, linewidth=0, color='#AA4499', alpha=0.3)
            ax.plot(epochs.times*1000, grp_avg9, label='ob baseline', color = '#44AA99')
            ax.fill_between(epochs.times*1000, grp_avg9-grp_sem9, grp_avg9+grp_sem9, linewidth=0, color='#44AA99', alpha=0.3)
            ax.plot(epochs.times*1000, grp_avg10, label='uu baseline', color = '#999933')
            ax.fill_between(epochs.times*1000, grp_avg10-grp_sem10, grp_avg10+grp_sem10, linewidth=0, color='#999933', alpha=0.3)
            ax.axvline(0, color='k')
            ax.axvspan(100, 150, alpha=0.5, color='gainsboro')
            ax.set_xlabel('Time (ms)', fontsize=18)
            ax.set_ylabel('Probability', fontsize=18)
            ax.xaxis.set_ticks(np.arange(-200,500,50))
            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
            ax.legend(['/ø/ prediction','/ɯ/ prediction','/ø/ baseline','/ɯ/ baseline'], prop = {'size':18}, loc='upper right')
            ax.set_title('Decoding probability of /u/ from /ø/ and /ɯ/', fontsize=20)
            # plt.savefig(dec_dir + '/grp/plots/grp_proba_ob_uu.png')
            plt.show()

# get ready to plot topography + evoked together
subjects = ['A0280','A0318','A0392','A0396','A0416','A0417']
subject = 'A0280'

for subject in subjects:
    epochs_fname = os.path.join(dec_dir, subject,'%s_biphon-epo.fif'%subject)
    epochs = mne.read_epochs(epochs_fname)


    plot_picks = ['meg']
    ts_args = dict(gfp=True, spatial_colors=True, ylim = dict(mag=[-120, 120]), time_unit='ms', titles=dict(mag='MEG'))  # ignore warnings about spatial colors
    topomap_args=dict(title='Single Subject Topography', time_unit='ms')

    # evo_kwargs['ylim'] = dict(mag=[-100, 120])
    epochs_A = epochs['position==1']


    # plot original data (averaged across epochs)
    fig = epochs_A.average(picks=plot_picks).plot_joint(times = np.array([0.075,.11]), picks=plot_picks, ts_args=ts_args, topomap_args=topomap_args)
    mne.viz.tight_layout()
    fig.savefig(dec_dir + '/grp/plots/single_subject_topo_%s.png'%subject, bbox_inches='tight')


### just evoked averages, no topo
## slice up epochs based on i and ah as AB and yih and y as X
# for subject in subjects:
#     epochs_i_AB_fname = os.path.join(dec_dir, 'grp', 'plots', '%s_biphon_i_AB-epo.fif'%subject)
#     epochs_ah_AB_fname = os.path.join(dec_dir,  'grp', 'plots', '%s_biphon_ah_AB-epo.fif'%subject)
#     epochs_y_X_fname = os.path.join(dec_dir,  'grp', 'plots', '%s_biphon_y_X-epo.fif'%subject)
#     epochs_yih_X_fname = os.path.join(dec_dir,  'grp', 'plots', '%s_biphon_yih_X-epo.fif'%subject)
#     epochs_i_AB = epochs['i_ah_AB==1']
#     epochs_ah_AB = epochs['ah_i_AB==1']
#     epochs_yih_X = epochs[idx_yih]
#     epochs_y_X = epochs[idx_y]
#     epochs_i_AB.save(epochs_i_AB_fname)
#     epochs_ah_AB.save(epochs_ah_AB_fname)
#     epochs_yih_X.save(epochs_y_X_fname)
#     epochs_y_X.save(epochs_yih_X_fname)
#
# vowel_labels = ['i_AB','ah_AB','y_X','yih_X']
# a=0
#
# for vowel_label in vowel_labels:
#     for subject in subjects:
#         epochs_fname = os.path.join(dec_dir, 'grp', 'plots', '%s_biphon_%s-epo.fif'%(subject, vowel_label))
#         epochs = mne.read_epochs(epochs_fname)
#         a=a+1
#     conglo_epochs = mne.concatenate_epochs([epochs_0,epochs_1,epochs_2,epochs_3])



epochs_i = epochs['i==1']
epochs_ah = epochs['ah==1']
epochs_yih = epochs['yih==1']
epochs_y = epochs['y==1']

epochs_i_AB = epochs['i_ah_AB==1']
epochs_ah_AB = epochs['ah_i_AB==1']
epochs_yih_X = epochs[idx_yih]
epochs_y_X = epochs[idx_y]

epochs_list = [epochs_i, epochs_ah, epochs_yih, epochs_y]
subject = 'A0280'


for subject in subjects:
    for epoch in epochs_list:
        plot_evo_kw = dict(picks='mag', spatial_colors=True, time_unit='ms', window_title='', titles='MEG', ylim=dict(mag=[-100,120]))
        fig = epoch.average().plot(**plot_evo_kw)
        fig.savefig(dec_dir + '/grp/plots/%s_evoked_%s.png'%(epoch.metadata['vowel_iso'].values[0], subject), bbox_inches='tight')


epochs_i_AB.average().plot(picks='mag', spatial_colors=True, gfp=True)
epochs_ah_AB.average().plot(picks='mag', spatial_colors=True, gfp=True)


epochs_yih_X.average().plot(picks='mag', spatial_colors=True, gfp=True)
epochs_y_X.average().plot(picks='mag', spatial_colors=True, gfp=True)



epochs_i_AB.average().plot()


# clean up
del epochs, epochs_clean
