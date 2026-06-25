import numpy as np
import json

all_res=[]
for atp in [0,0.3,0.5,0.7,0.9,1]:
    day_res=[]
    for day in range(1,6):

        path=r'E:\BaiduNetdiskDownload\.idea\majun\adapt_EEG\results\adapt_true\adapt='+str(atp)+'\day='+str(day)
        from os.path import join

        res=[]
        for subj in range(25):
            subj_acc=[]
            for fold in range(1,11):
                with open(join(path,r'test_s'+str(subj+1)+'_f'+str(subj)+'_'+str(fold)+'.json'), 'r') as f:
                    a=json.load(f)
                subj_acc.append(1-a['misclass'])
            acc=np.average(subj_acc)
            res.append(acc)
        day_res.append([day,res])
    all_res.append([atp,day_res])


# res0_1=all_res[0][1][0][1]
# res0_2=all_res[0][1][1][1]
# res0_3=all_res[0][1][2][1]
# res0_4=all_res[0][1][3][1]
# res0_5=all_res[0][1][4][1]
#
# res03_1=all_res[1][1][0][1]
# res03_2=all_res[1][1][1][1]
# res03_3=all_res[1][1][2][1]
# res03_4=all_res[1][1][3][1]
# res03_5=all_res[1][1][4][1]

res=[]
aver=[]
for i in range(30):
    res.append(all_res[int(i/5)][1][i%5][1])
    print(np.average(all_res[int(i/5)][1][i%5][1]))
# print(np.average(res1))
resmat=np.array(res)

# import scipy.io as sio
# sio.savemat('resmat.mat',{'resmat':resmat})