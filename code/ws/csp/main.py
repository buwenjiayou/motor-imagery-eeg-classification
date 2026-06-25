#SHU Dataset for CSP algorithm
# sub_id, input subject id 1 to 25
# session, input session id 1 to 5

from ws import train_test
import os
import scipy.io as sio
import numpy as np



res=[]
for sub_id in range(1,26):
    for session in range(1,6):
        data_path=os.path.abspath(os.path.join(os.getcwd(),"../.."))
        wst_accuracy=train_test(sub_id,session,data_path,k_fold=10)
        print('CSP acc is: ',wst_accuracy)
        res.append(wst_accuracy)

sio.savemat('results.mat',{'res':res})
print(np.average(res))




