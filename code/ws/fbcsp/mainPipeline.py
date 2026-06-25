#SHU Dataset for FBCSP algorithm
#Original algorithm link is  https://fbcsptoolbox.github.io/publications


from bin.MLEngine import MLEngine
import scipy.io as sio
import os

if __name__ == "__main__":

    result=[]
    for subj in range(1,26):
        for session in range(1,6):
            '''Example for loading SHU Dataset '''
            dataset_details={
                'data_path' : os.path.abspath(os.path.join(os.getcwd(),"../..")),
                'file_to_load': [subj,session],
                'ntimes': 1,
                'kfold':10,
                'm_filters':4,
                'window_details':{'tmin':0.5,'tmax':4-0.004}
            }

            ML_experiment = MLEngine(**dataset_details)
            result.append(ML_experiment.experiment())
sio.savemat('result.mat',{'res':result})