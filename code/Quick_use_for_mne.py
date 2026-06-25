# This is a mne.raw create method, users can perform operations on raw files as required.

from ws.csp.data_preprocess import *
import os

subj=1
session=1


data,labels=get_data(subj,session,'')
raw=create_raw(data,labels)