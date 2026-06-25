#First run get_data to transform the SHU Dataset to fit the FBCNet toolbos
#Then run classify.cv to train and test SHU dataset in within-session classification



import os
import numpy as np
import scipy.io as sio
import shutil
from os.path import join

data_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
save_path= join(data_path,r'ws/FBCNet/data/shu/originalData')
if not os.path.isdir(save_path):
    os.makedirs(save_path)
new_id=1
for subj in range(1,26):
    for session in range(1,6):
        # load_path=join(data_path,'SHU_Dataset','Subj_'+str(subj).zfill(2)+'session_'+str(session)+'.mat')
        # da=sio.loadmat(load_path)
        da=sio.loadmat(join(data_path,'SHU_Dataset','sub-'+str(subj).zfill(3)+'_ses-'+str(session).zfill(2)+'_task_motorimagery_eeg.mat'))

        data=da['data']
        labels=np.ravel(da['labels'])
        Edata=data[:50]
        Elabels=labels[:50]
        Tdata=data[50:]
        Tlabels=labels[50:]
        sio.savemat(join(save_path,'S'+str(new_id).zfill(3)+'E.mat'),{'data':Edata,'labels':Elabels})
        sio.savemat(join(save_path,'S'+str(new_id).zfill(3)+'T.mat'),{'data':Tdata,'labels':Tlabels})
        new_id+=1

