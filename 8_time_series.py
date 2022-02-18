
'''...................... Launch IPython with eelbrain ......................'''

import numpy as np
import pandas as pd
import mne, pickle, eelbrain
from os import chdir
import os.path as op
# from surfer import Brain



'''..................... directory, date, output folder .....................'''

# set up current directory
root = '/Volumes/bin_battuta/biphon'
subjects_dir = op.join(root,'mri')
chdir(root)

# date of analysis
date = '24March2020'

# output folder path
output = op.join(root, 'grp/%s/time_series_left' %date)




'''.................... lists of subjects and conditions ....................'''

subjects = ['A0280','A0318','A0392','A0396','A0416','A0417']
conditions = ['nonnative']
# vowels = [['i1_A','i2_A'],['u1_A','u2_A'],['ah1_A','ah2_A'],['ae1_A','ae2_A'],
#             ['i1_B','i2_B'],['u1_B','u2_B'],['ah1_B','ah2_B'],['ae1_B','ae2_B'],
#             ['i1_X','i2_X'],['u1_X','u2_X'],['ah1_X','ah2_X'],['ae1_X','ae2_X'],
#             ['yih1_A','yih2_A'],['yih1_B','yih2_B'],['yih1_X','yih2_X'],
#             ['y1_A','y2_A'],['y1_B','y2_B'],['y1_X','y2_X'],
#             ['uu1_A','uu2_A'],['uu1_B','uu2_B'],['uu1_X','uu2_X'],
#             ['ob1_A','ob2_A'],['ob1_B','ob2_B'],['ob1_X','ob2_X']]
#
# vowels = ['i1_A_i2_A', 'u1_A_u2_A', 'ah1_A_ah2_A', 'ae1_A_ae2_A',
#             'i1_B_i2_B', 'u1_B_u2_B', 'ah1_B_ah2_B', 'ae1_B_ae2_B',
#             'i1_X_i2_X', 'u1_X_u2_X', 'ah1_X_ah2_X', 'ae1_X_ae2_X']

# vowels = ['i1_A_i2_A', 'u1_A_u2_A', 'ah1_A_ah2_A', 'ae1_A_ae2_A',
#             'i1_B_i2_B', 'u1_B_u2_B', 'ah1_B_ah2_B', 'ae1_B_ae2_B',
#             'i1_X_i2_X', 'u1_X_u2_X', 'ah1_X_ah2_X', 'ae1_X_ae2_X','i1_A_i2_A','u1_A_u2_A','ah1_A_ah2_A','ae1_A_ae2_A','y1_A_y2_A','yih1_A_yih2_A','ob1_A_ob2_A','uu1_A_uu2_A',
#             'i1_B_i2_B','u1_B_u2_B','ah1_B_ah2_B','ae1_B_ae2_B','y1_B_y2_B','yih1_B_yih2_B','ob1_B_ob2_B','uu1_B_uu2_B',
#             'i1_X_i2_X','u1_X_u2_X','ah1_X_ah2_X','ae1_X_ae2_X','y1_X_y2_X','yih1_X_yih2_X','ob1_X_ob2_X','uu1_X_uu2_X']

# vowels = ['i1_A_i2_A', 'u1_A_u2_A', 'ah1_A_ah2_A', 'ae1_A_ae2_A',
#             'y1_A_y2_A','yih1_A_yih2_A','ob1_A_ob2_A','uu1_A_uu2_A']
#
# vowels = ['i1_B_i2_B', 'u1_B_u2_B', 'ah1_B_ah2_B', 'ae1_B_ae2_B',
#             'y1_B_y2_B','yih1_B_yih2_B','ob1_B_ob2_B','uu1_B_uu2_B']

vowels = ['i1_X_i2_X', 'u1_X_u2_X', 'ah1_X_ah2_X', 'ae1_X_ae2_X',
            'y1_X_y2_X','yih1_X_yih2_X','ob1_X_ob2_X','uu1_X_uu2_X']




'''.................. get fsaverage vertices for morphing ...................'''

print('Reading in source space...')
subject_to = 'fsaverage'
fs = mne.read_source_spaces('mri/fsaverage/bem/fsaverage-ico-4-src.fif')
vertices_to = [fs[0]['vertno'], fs[1]['vertno']]

'''......................... read in morphed stcs ...........................'''

stcs,subjectlist,conditionlist,vowellist = [],[],[],[]

for subject in subjects:
    print('Reading in source estimates from Subject %s...' %subject)
    for condition in conditions:
        for vowel in vowels:
            tmp = mne.read_source_estimate('evoked/%s/%s_%s_%s_morphed-lh.stc' %(subject,subject,condition,vowel),subject='fsaverage')
            stcs.append(tmp)
            subjectlist.append(subject)
            conditionlist.append(condition)
            vowellist.append(vowel)
            del tmp




'''................ create dataset with stcs, subjs, conds ..................'''

print ('Creating dataset...')
ds = Dataset()
# asso = [str.split(i,'_')[0] for i in conditionlist]
# comp = [str.split(i,'_')[1] for i in conditionlist]
vowel = vowellist

print ('Reading in stcs...')
ds['stcs'] = load.fiff.stc_ndvar(stcs, subject='fsaverage', src='ico-4', subjects_dir=subjects_dir, parc='aparc') # parcellating source space
ds['Subject'] = Factor(subjectlist,random=True)
ds['Condition'] = Factor(conditionlist)
# ds['Association'] = Factor(asso)
# ds['Composition'] = Factor(comp)
ds['Vowel'] = Factor(vowel)
src_reset = ds['stcs']




'''...................... plot time courses by region .......................'''

# language network regions of interest
# regions = ['IFG-lh', 'vmPFC-lh', 'ATL+MTL-lh', 'PTL-lh', 'TPJ-lh','V1-lh']
regions = ['superiortemporal-lh']
xticks=np.arange(-0.2,0.6,0.1)
plot_by = 'Vowel' # Condition, Association, Composition

for region in regions:

    print('Plotting time series at %s...' %region)
    ds['srcm'] = src_reset
    src_region = src_reset.sub(source=region) # subset language network region data
    ds['srcm'] = src_region # assign this back to the ds
    timecourse = src_region.mean('source')

    activation = eelbrain.plot.UTSStat(timecourse, plot_by, ds=ds, match='Subject', legend=None, error=None,
        xlabel='Time (ms)', ylabel='Activation (dSPM)', tight=True, xlim=(0,0.6), frame=None, title='Time Series at %s' %region)
    activation._axes[0].set_xticks(xticks)
    activation._axes[0].lines[0].set_lw(w=2)
    # activation._axes[0].lines[1].set_ls(ls='--')
    # activation._axes[0].lines[1].set_lw(w=2)
    activation._axes[0].legend(labels=ds['Vowel'], loc='upper right')
    activation._axes[0].legend()
    activation.set_ylim(-1,1)
    activation._axes[0].set_xticks(xticks)
    activation._axes[0].axvline(linewidth = '2', x=0, color='k', linestyle='--')
    activation.save(op.join(output, 'ts_%s_max3.jpg' %region), dpi=1200, quality=95, optimize=True, format='jpg', transparent=True)
    activation.close()

# @ colour scheme
# position  (5,6,7): '#DDAA33','#AA3377','#004488'
# association (high, low): '#CCBB44','#AA4499'
# composition (sent, list): '#332288','#117733'
# composition*position: '#EE7733','#CC3311','#EE3377','#0077BB', '#33BBEE','#009988'
# condition (high_sent, high_list, low_sent, low_list): 'royalblue', 'orange', 'lightseagreen', 'tomato'
