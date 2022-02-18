# eelbrain is convenient here to create dataset
import mne, os, glob # launch ipython with eelbrain profile
import eelbrain
import pandas as pd
import numpy as np
import time

# set up current directory
# root = '/Volumes/bin_battuta/biphon'
root = '/Volumes/hecate/biphon'
os.chdir(root)
os.environ["SUBJECTS_DIR"] = os.path.join(root,'mri')

# # reading in semantic association measures // e.g. here is where you might Formant measures // maybe skip this if you have all the info (for variables) you need
# with open('Association/similarity_comparison.csv') as f:
#     rows = (line.split(',') for line in f) # read in each line, separated by comma
#     glove_dict = {row[0]:row[1].strip('\n') for row in rows} # create dictionary and strip away useless bits
#     # row[0] is the key, row[1] is the value

''' --------------- make LMM output --------------- '''
subjects = ['A0280','A0318','A0392','A0396','A0416','A0417']

print('Reading in source space and parcellation...')
parc = mne.read_labels_from_annot('fsaverage', 'aparc', hemi='lh') # parcellation your brain, e.g. aparc, PALS_B12_Brodmann
label = [i for i in parc if i.name.startswith('superiortemporal-lh')][0] #

# parc = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', hemi='lh') # parcellation your brain, e.g. aparc, PALS_B12_Brodmann
# label = [label for label in parc if label.name == 'L_A1_ROI-lh', 'L_A4_ROI-lh'][0] #

src_fname = os.path.join(root, 'mri/fsaverage/bem/fsaverage-ico-4-src.fif')
src = mne.read_source_spaces(src_fname)


for subject in subjects:
    print ('------------------------')
    print ('Preparing Subject %s....' %subject)
    # stc_fname = os.path.join(root, 'stc/%s/%s_biphon_stc_epochs.stc.npy' %(subject,subject))
    # stc_epochs = np.load(stc_fname, allow_pickle=True)

    print ('Reading in trial info...')
    trialInfo_fname = os.path.join(root, 'meg/%s/%s_biphon_rej_trialinfo.csv' %(subject,subject))
    trialInfo = pd.read_csv(trialInfo_fname)
    # declare several empty lists that is necessary for my dataset
    dSPMList, subjectList, itemList, orderList, vowelList = [], [], [], [], []

    for index, row in trialInfo.iterrows():
        print (index, row['subject'], row['vowel_id'], row['trial_order'], row['trialid']) # sanity check printing
        ItemNum = row['trialid']
        vowels = row['vowel_id']
        order = row['trial_order']
        # /Volumes/bin_battuta/biphon/stc/A0416/A0416_1-lh.stc
        print ('Reading in epoch STC from %s...' %subject)
        stc = mne.read_source_estimate(root + '/stc/%s/%s_%s'%(subject,subject,str(index)) + '-lh.stc') #this just reads in the left hemipshere stc
        print ('Extracting STG and STS time courses from trial %s...' %index)
        dSPM = mne.extract_label_time_course(stc, label, src=src, mode='mean')[0]
        t_min = 300 #set your time window in ms here (IMPORTANT!!!! This starts from the beginning of the array so make sure the number accounts for a baseline)
        t_max = 350 #set your time window in ms here
        dSPM = dSPM[t_min:t_max] #set your time window in ms here
        average_dSPM = float(sum(dSPM)/len(dSPM))
        dSPMList.append(float(average_dSPM))
        subjectList.append(subject)
        itemList.append(ItemNum)
        vowelList.append(vowels)
        orderList.append(order)
        del stc
        print ('Done.')

    #make dataset to output as text for input to R. I did this in an eelbrain dataset out of laziness, a more appropriate dataframe like pandas could also be used.
    ds = Dataset()
    ds['dSPM'] = Factor(dSPMList)
    ds['Subject'] = Factor(subjectList, random=True)
    ds['Item'] = Factor(itemList)
    ds['Vowel'] = Factor(vowelList)
    ds['Order'] =  Factor(orderList)
    #%store ds.as_table() >> biphon_lmer_20191212.csv
    #this requires ipython and needs to be copy-pasted into the ipython shell directly. will make a space-separated file. you'll need to replace those spaces with commas to read it into R as a csv
    ds = pd.DataFrame(ds)
    trialInfo = pd.concat([trialInfo,ds], axis=1)
    trialInfo.to_csv(os.path.join(root, 'meg/%s/%s_biphon_lmer_dSPM.csv'%(subject,subject)))

# Concatenate all lmer info files
all_filenames = []
timestr = time.strftime("%Y%m%d")
for subject in subjects:
    lmer_dSPM_fname = os.path.join(root, 'meg/%s/%s_biphon_lmer_dSPM.csv' %(subject,subject))
    print(lmer_dSPM_fname)
    all_filenames.append(lmer_dSPM_fname)
print('%s CSV files found and extracted'%len(all_filenames))
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv(os.path.join(root, 'meg/lmer/biphon_combined_lmer_%s.csv'%timestr), index=True, encoding='utf-8-sig')


# %store ds.as_table() >> biphon_lmer_20191212.csv
#this requires ipython and needs to be copy-pasted into the ipython shell directly. will make a space-separated file. you'll need to replace those spaces with commas to read it into R as a csv
