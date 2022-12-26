function [X,label,p] = load_dataset(id)
addpath datasets
%% Load data
if id==1    
    data=load('winequality-red.dat');
    p=6;
    X=data(:,1:end-1);
    label=data(:,end)';
elseif id==2
    data=load('banana.dat');
    p=2;
    X=data(:,1:end-1);
    label=data(:,end)';
end

%% Data normalization
X=(X-min(X))./(max(X)-min(X));

end

