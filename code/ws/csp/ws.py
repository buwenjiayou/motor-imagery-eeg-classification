import numpy as np
from sklearn.model_selection import StratifiedKFold
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from data_preprocess import get_data,mnebandFilter


def train_test(sub_id,test_session,data_path,k_fold=10):
    data,labels=get_data(sub_id,test_session,data_path)
    data=mnebandFilter(data,labels,3,35)
    cv = StratifiedKFold(k_fold, shuffle=False)  # k-fold StratifiedKFold      ShuffleSplit
    cv_split = cv.split(data, labels)
    scores = []
    for i, j in cv_split:  # k-fold
        trainData = data[i]
        trainLabels = labels[i]
        valData = data[j]
        valLabels = labels[j]
        csp = CSP(n_components=10, reg=None, log=False, norm_trace=False)
        trainFeature = csp.fit_transform(trainData, trainLabels)
        lda = LinearDiscriminantAnalysis()
        lda.fit(trainFeature, trainLabels)
        valFeature = csp.transform(valData)
        scores.append([lda.score(valFeature, valLabels)])
    aver_score=np.average(scores)
    return aver_score
