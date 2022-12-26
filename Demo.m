%% Load dataset
id=1; %1 or 2
[X,label,p] = load_dataset(id);
n=size(X,1);
m=size(X,2);

%% Parameter setting
N=2;
M=5;
s=-5;
eta=1.1;

rng(1);
%% CAPKM++2.0
[resulting_label,obj_value]=capkm_pro(X,p,n,m,eta,s,M,N);
%% It is also optional to use default parameters
%[resulting_label,obj_value]=capkm_pro(X,p,n,m,eta,s);
%[resulting_label,obj_value]=capkm_pro(X,p,n,m);