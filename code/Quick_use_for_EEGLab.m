eeglab;

subj=1;
session=1;

load(strcat('.\SHU_dataset\sub-',num2str(subj,'%03d'),'_ses-',num2str(session,'%02d'),'_task_motorimagery_eeg.mat'));
data=permute(data,[2,3,1]);
save('data.mat','data');
EEG = pop_importdata('dataformat','matlab','nbchan',0,'data','data.mat','srate',250,'pnts',0,'xmin',0,'chanlocs','location.ced');
pop_eegplot( EEG, 1, 1, 1);