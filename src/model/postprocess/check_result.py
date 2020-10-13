import os
import os.path as osp 
import shutil
import tqdm

data_dir = 'data'
date = 'Oct12_1'
result_dir = f'result/resunet++/{date}/visualize_val'

easy_test_dir = osp.join(data_dir, 'test_easy')
hard_test_dir = osp.join(data_dir, 'test_hard')
org_test_dir = osp.join(data_dir, 'test_images')
unk_test_dir = osp.join(data_dir, 'unk_test')

result_img = os.listdir(result_dir)
easy_train = os.listdir(osp.join(data_dir, 'train_easy'))
easy_test = os.listdir(easy_test_dir)
hard_test = os.listdir(hard_test_dir)
org_test = os.listdir(org_test_dir)
org_train = os.listdir(osp.join(data_dir, 'train_images'))



for org_img in tqdm.tqdm(easy_test):
    if org_img in hard_test:
        print(org_img)
    

