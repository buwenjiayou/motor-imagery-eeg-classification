'''
Preprocessor for SHu Dataset.
First run this file, then run train_base and train adapt
Original algorithm link is  https://github.com/zhangks98/eeg-adapt
'''
import argparse
from os.path import join as pjoin
import h5py
import numpy as np
import torchvision.datasets.utils
from scipy.io import loadmat
from scipy.signal import decimate
from tqdm import tqdm
import os
import scipy.io as sio





def get_data(subj):
    file_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))



    data=np.empty((0,32,1000))
    labels=np.empty(0)
    for session in range(1,6):
        # da=sio.loadmat(pjoin(file_path,'SHU_Dataset','Subj_'+str(subj).zfill(2)+'session_'+str(session)+'.mat'))
        da=sio.loadmat(pjoin(file_path,'SHU_Dataset','sub-'+str(subj).zfill(3)+'_ses-'+str(session).zfill(2)+'_task_motorimagery_eeg.mat'))
        data=np.vstack((data,da['data']))
        labels=np.hstack((labels,np.ravel(da['labels'])))

    X=data
    Y=(np.ravel(labels)-1).astype(np.int64)
    return X, Y


with h5py.File('data\SHU_data.h5', 'w') as f:
    for subj in tqdm(range(1, 26)):
        X, Y = get_data( subj)
        f.create_dataset('s' + str(subj) + '/X', data=X)
        f.create_dataset('s' + str(subj) + '/Y', data=Y)