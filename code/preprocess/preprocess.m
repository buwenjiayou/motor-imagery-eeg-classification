load Unity×óÊÖÓÒÊÖ2018_03_31_194531202.mat

trail=zeros(20,1000,32);
data=BCI.info.data;
lb=BCI.info.label;
labels=zeros(1,20);
buf=1;

for i =1:20
        da=data(1:32,:,i);
        da=da*(2.5*10^6/(49.9)^2/7864320);
        trail(buf,:,:)=detrend(da');
        labels(buf)=lb(i);
        buf=buf+1;
end 
label=labels';

save ('data.mat','trail','label')