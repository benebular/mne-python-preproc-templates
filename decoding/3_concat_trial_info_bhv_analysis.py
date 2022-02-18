## helper script for concatenating trial info post epoch rejection for behavioral data in decoding analysis
# author: Ben Lang, blang@ucsd.edu

import mne, os, glob # launch ipython with eelbrain profile
import pandas as pd
import numpy as np
import time

# set up current directory
root = '/Volumes/hecate/biphon'
os.chdir(root)
os.environ["SUBJECTS_DIR"] = os.path.join(root,'mri')
subjects = ['A0280','A0318','A0392','A0396','A0416','A0417']

# Concatenate all lmer info files
all_filenames = []
timestr = time.strftime("%Y%m%d")
for subject in subjects:
    lmer_dSPM_fname = os.path.join(root, 'dec/%s/%s_biphon_rej_trialinfo.csv' %(subject,subject))
    print(lmer_dSPM_fname)
    all_filenames.append(lmer_dSPM_fname)
print('%s CSV files found and extracted'%len(all_filenames))
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv(os.path.join(root, 'dec/grp/biphon_combined_bhv_dec_%s.csv'%timestr), index=True, encoding='utf-8-sig')
