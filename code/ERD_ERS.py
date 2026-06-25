import mne
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import CSP

def create_raw(data,labels,info,freq=250):
    labels=np.ravel(labels)
    ch_names=info.ch_names
    si,sj,sk=data.shape
    da=data.transpose(1,0,2)
    da=da.reshape(sj,si*sk)
    llen=data.shape[0]
    event=np.zeros((llen,3))
    info = mne.create_info(
        ch_names=ch_names,
        ch_types="eeg",  # channel type
        sfreq=freq  # frequency
    )
    raw = mne.io.RawArray(da, info)  # create raw
    for i in range(llen):
        event[i,0]=i*sk
        event[i,2]=labels[i]
    event=event.astype(int)
    raw.info['events']=event
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    return raw,event


def smooth2nd(x, M=80):  ## data smooth
    K = round(M / 2 - 0.1)
    lenX = len(x)
    if lenX < 2 * K + 1:
        print('Error')
    else:
        y = np.zeros(lenX)
        for NN in range(0, lenX, 1):
            startInd = max([0, NN - K])
            endInd = min(NN + K + 1, lenX)
            y[NN] = np.mean(x[startInd:endInd])
    return (y)

def get_erders(datas):
    data=datas.transpose(1,2,0)
    data=data*data
    C3=np.sum(data[12,:,:],axis=1)
    C4=np.sum(data[14,:,:],axis=1)
    C3=C3/len(datas)
    C4=C4/len(datas)
    C3=smooth2nd(C3,80)
    C4=smooth2nd(C4,80)
    C3mean=np.average(C3[0:125])
    C4mean=np.average(C4[0:125])
    C3=((C3-C3mean)/C3mean)*1
    C4=((C4-C4mean)/C4mean)*1
    return C3,C4

def check_data(data):
    resdata=[]
    for i in range(len(data)):
        dats=data[i]
        s=np.sum(dats>100)
        if s ==0:
            resdata.append(dats)
    resdata=np.array(resdata)
    return resdata


data_all=np.empty((0,32,1000))
labels_all=np.empty((0))
for subj in range(1,26):
    for se in range(1,6):
        path=r'.\SHU_Dataset\Subj_{}session_{}.mat'.format(str(subj).zfill(2),str(se))

        info = mne.create_info(
            ch_names=["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
                    "FC6", "Cz", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2",
                    "CP5", "CP6", "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz",
                    "O1", "O2"],
            ch_types='eeg', sfreq=250)

        da=sio.loadmat(path)
        data=da['data']
        labels=np.ravel(da['labels'])

        raw, event = create_raw(data, labels, info)
        raw.filter(8, 30, fir_design='firwin', skip_by_annotation='edge')
        event_id = dict(left=1, right=2)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, ecg=False, exclude='bads')
        epochs = mne.Epochs(raw, event, event_id, 0, 3.996, proj=True, picks=picks, baseline=None, preload=True)
        data=epochs.get_data()
        data_all=np.vstack((data_all,data))
        labels_all=np.hstack((labels_all,np.ravel(labels)))


data=data_all
labels=labels_all



data_left=[]
data_right=[]
for i in range(len(data)):
    if labels[i]==1:
        data_left.append(data[i])
    else:
        data_right.append(data[i])
data_left=np.array(data_left)
data_right=np.array(data_right)

data_left=check_data(data_left)



LC3,LC4=get_erders(data_left)
RC3,RC4=get_erders(data_right)

plt.subplot(211)
plt.title('ERD/ERS on channel C3 and Channel C4')
plt.plot(LC3,label='left')
plt.plot(RC3,label='right')
plt.ylabel('Channel C3')
plt.xlabel('time/ms')
plt.legend(loc=0)
plt.subplot(212)
plt.plot(LC4,label='left')
plt.plot(RC4,label='right')
plt.ylabel('Channel C4')
plt.xlabel('time/ms')
plt.legend(loc=0)
plt.show()





