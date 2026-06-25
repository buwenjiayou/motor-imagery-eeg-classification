#First run get_data to transform the SHU Dataset to fit the FBCNet toolbos
#Then run classify.cv to train and test SHU dataset in within-session classification



import os
import numpy as np
import scipy.io as sio
import shutil
from os.path import join

data_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
save_path= join(data_path,r'cs/FBCNet/data/shu/originalData')
if not os.path.isdir(save_path):
    os.makedirs(save_path)


test_session=2	#set different test session 2 to 5
for subj in range(1,26):
    # load_path=join(data_path,'SHU_Dataset','Subj_'+str(subj).zfill(2)+'session_1.mat')
    da=sio.loadmat(join(data_path,'SHU_Dataset','sub-'+str(subj).zfill(3)+'_ses-'+str(1).zfill(2)+'_task_motorimagery_eeg.mat'))
    # da=sio.loadmat(load_path)
    data=da['data']
    labels=np.ravel(da['labels'])
    Edata=data
    Elabels=labels

    # load_path = join(data_path, 'SHU_Dataset', 'Subj_' + str(subj).zfill(2) + 'session_' + str(test_session) + '.mat')
    da=sio.loadmat(join(data_path,'SHU_Dataset','sub-'+str(subj).zfill(3)+'_ses-'+str(test_session).zfill(2)+'_task_motorimagery_eeg.mat'))
    # da=sio.loadmat(load_path)
    data=da['data']
    labels=np.ravel(da['labels'])
    Tdata=data
    Tlabels=labels

    sio.savemat(join(save_path,'S'+str(subj).zfill(3)+'E.mat'),{'data':Edata,'labels':Elabels})
    sio.savemat(join(save_path,'S'+str(subj).zfill(3)+'T.mat'),{'data':Tdata,'labels':Tlabels})
