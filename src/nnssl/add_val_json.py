import os
import json
import shutil
import random
random.seed(616)
# adds test data to train data and updates
# {
    # "train": [
    #     "Dataset001_MerlinTrain__Dataset001_MerlinTrain__AC423c68a",
    #     "Dataset001_MerlinTrain__Dataset001_MerlinTrain__AC423ccce",
nii_path = ''
src_path = '/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_preprocessed/Dataset001_MerlinTrain/splits_final_tr.json'
dest_path = '/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_preprocessed/Dataset001_MerlinTrain/splits_final.json'

train_path = '/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_preprocessed/Dataset001_MerlinTrain/nnsslPlans_onemmiso/Dataset001_MerlinTrain/Dataset001_MerlinTrain'
to_add_path = '/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_preprocessed/Dataset001_MerlinTrain/nnsslPlans_onemmiso/Dataset001_MerlinTrain/Dataset003_MerlinTest'
os.makedirs(to_add_path, exist_ok=True)
test_path = '/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_preprocessed/Dataset003_MerlinTest/nnsslPlans_onemmiso/Dataset003_MerlinTest/Dataset003_MerlinTest'

files = os.listdir(test_path)
files = random.sample(files, 24)
print(files)

string1 = 'Dataset001_MerlinTrain'
string2 = 'Dataset003_MerlinTest'
vals = []
for f in files:
    st = string1 + '__' + string2 + '__' + f
    vals.append(st)
    src_p = os.path.join(test_path, f)
    dest_p = os.path.join(to_add_path, f)
    shutil.copytree(src_p, dest_p)


with open(src_path, 'rb') as f:
    json_data = json.load(f)

json_data['val'] = vals
#print(json_data)
with open(dest_path, 'w') as f:
    json.dump(json_data, f)

# {
#     "collection_index": 1,
#     "collection_name": "Dataset001_MerlinTrain",
#     "datasets": {
#         "Dataset001_MerlinTrain": {
#             "dataset_index": "Dataset001_MerlinTrain",
#             "dataset_info": {},
pretrain_train_path = '/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_preprocessed/Dataset001_MerlinTrain/pretrain_data__onemmiso_tr.json'
pretrain_test_path = '/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_preprocessed/Dataset003_MerlinTest/pretrain_data__onemmiso.json'
pretrain_dest_path = '/cluster/projects/bwanggroup/sumin/nnssl_data/nnssl_preprocessed/Dataset001_MerlinTrain/pretrain_data__onemmiso.json'

with open(pretrain_train_path, 'rb') as f:
    json_data_train = json.load(f)
with open(pretrain_test_path, 'rb') as f:
    json_data_test = json.load(f)

obtain_from_test = {}
# test_cases = list(json_data_test['datasets']['Dataset003_MerlinTest']['subjects'].keys())
# import pdb;pdb.set_trace()
# for test in test_cases:
#     if test in files:
#         snippet = json_data_test['datasets']['Dataset003_MerlinTest']['subjects'][test]
#         snippet['sessions']['ses-DEFAULT']['images'][0]['image_path'] = snippet['sessions']['ses-DEFAULT']['images'][0]['image_path'].replace('nnssl_preprocessed/Dataset003_MerlinTest', 'nnssl_preprocessed/Dataset001_MerlinTrain')
#         #obtain_from_test.append(snippet)
#         obtain_from_test[test] = snippet
# #import pdb;pdb.set_trace()
# #json_data_test['datasets']['Dataset003_MerlinTest']['subjects'] = obtain_from_test

# json_data_train['datasets']['Dataset001_MerlinTrain']['subjects'].update(obtain_from_test)
# #import pdb; pdb.set_trace()
json_data_train['datasets'].update(json_data_test['datasets'])

with open(pretrain_dest_path, 'w') as f:
    json.dump(json_data_train, f, indent=4)
print(pretrain_dest_path)
