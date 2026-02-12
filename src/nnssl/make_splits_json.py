import os
import numpy as np
import pandas as pd
import json
import glob

dest_path = '/scratch/kmin940/nnssl_data/nnssl_preprocessed/Dataset001_FLARE25_CT/splits_final.json'
b2nd_path = '/scratch/kmin940/nnssl_data/nnssl_preprocessed/Dataset001_FLARE25_CT/nnsslPlans_onemmiso/Dataset001_FLARE25_CT/Dataset001_FLARE25_CT'


# randomly select 10 cases for validation
files = os.listdir(b2nd_path)
print(f'Total files before filtering: {len(files)}')
files_remove = [x for x in files if len(glob.glob(os.path.join(b2nd_path, x, '*', "*"))) <2]
print(f'Files to remove (less than 2 subdirs): {len(files_remove)}')
# remove these directories in files_remove with os.remove
for f in files_remove:
    full_path = os.path.join(b2nd_path, f)
    if os.path.isdir(full_path):
        os.system(f'rm -dfr {full_path}')
        #os.rmdir(full_path)
        print(f'Removed directory: {full_path}')
    else:
        print(f'Not a directory, cannot remove: {full_path}')
files = [f for f in files if f not in files_remove]


print(f'Total files after filtering: {len(files)}')
#files = [f[:-6] for f in files if f.endswith('.b2nd')]
#files = [f for f in files if f.endswith('.b2nd')]
files = ['Dataset001_FLARE25_CT__Dataset001_FLARE25_CT__' + x for x in files]
print(f'Total files: {len(files)}')
# exclude = [
#     '10302__12052022_BrnoKrios_Arctis_grid_newGISc_Position_4.b2nd', # shape mismatch
#     '10442__24aug01b_Position_7_4_denoised.b2nd', # shape mismatch
#     '10444__24apr23a_Position_29.b2nd', # shape mismatch
#     '10444__24apr23a_Position_3.b2nd'
# ]
# exclude = ['Dataset001_FLARE25_CT__Dataset001_FLARE25_CT__' + x for x in exclude]
# files = [f for f in files if f not in exclude]
files = [x.split('.b2nd')[0] for x in files]
print(f'Files after exclusion: {len(files)}')

np.random.seed(1234)
val_cases = np.random.choice(files, size=5, replace=False).tolist()
train_cases = [f for f in files if f not in val_cases]

splits = {
    "train": train_cases,
    "val": val_cases
}
print(f'train: {len(train_cases)}, val: {len(val_cases)}')
with open(dest_path, 'w') as f:
    json.dump(splits, f, indent=4)
# print(f'Saved to {dest_path}')

# Removed directory: /scratch/kmin940/nnssl_data/nnssl_preprocessed/Dataset001_FLARE25_CT/nnsslPlans_onemmiso/Dataset001_FLARE25_CT/Dataset001_FLARE25_CT/autoPET_fdg_1253499c80_10-07-2005-NA-PET-CT_Ganzkoerper__primaer_mit_KM-50242_0000
# Removed directory: /scratch/kmin940/nnssl_data/nnssl_preprocessed/Dataset001_FLARE25_CT/nnsslPlans_onemmiso/Dataset001_FLARE25_CT/Dataset001_FLARE25_CT/autoPET_fdg_4776e75543_04-01-2007-NA-PET-CT_Ganzkoerper__primaer_mit_KM-74749_0000
# Removed directory: /scratch/kmin940/nnssl_data/nnssl_preprocessed/Dataset001_FLARE25_CT/nnsslPlans_onemmiso/Dataset001_FLARE25_CT/Dataset001_FLARE25_CT/autoPET_fdg_d626611daf_11-29-2002-NA-PET-CT_Ganzkoerper__primaer_mit_KM-88747_0000
# Removed directory: /scratch/kmin940/nnssl_data/nnssl_preprocessed/Dataset001_FLARE25_CT/nnsslPlans_onemmiso/Dataset001_FLARE25_CT/Dataset001_FLARE25_CT/autoPET_fdg_f24f3ce1da_09-14-2001-NA-PET-CT_Ganzkoerper__primaer_mit_KM-27130_0000
# Removed directory: /scratch/kmin940/nnssl_data/nnssl_preprocessed/Dataset001_FLARE25_CT/nnsslPlans_onemmiso/Dataset001_FLARE25_CT/Dataset001_FLARE25_CT/psma_d67e027e14323000_2019-09-02_0000