function [label,gbest]=capkm_pro(X,p,n,m,eta,s,M,N)
if nargin<8
    % default parameters
    M=5;
    N=2;
end
if nargin<6
    % default parameters
    eta=1.1;
    s=-5;
end

tmax=1000;
w=1;beta1=1;beta2=1;
gbest=1e10;
gbest_pre=1e10;
pbestx=zeros(n,N*p);
pbest=1e10*ones(1,N);
gbestx=zeros(n,N*p);
s_not_too_small=1;
tic
for t=1:tmax
    %% generate the centers randomly, and assign weights
    [W2,pbest,pbestx,gbest,gbestx]=generate_centers_and_assign_w_initial_pbest_gbest(X,n,p,N,pbest,pbestx,gbest,gbestx);
    it=0;
    count_gli=0;
    v=2*rand(n,N*p)-1;
    if s_not_too_small
        while true
            %% Multiple models for scattered local search and determining the individual best.
            for nn_count=1:N
                W=W2(:,(nn_count-1)*p+1:nn_count*p);
                norm_flg=1;

                for l=1:p
                    for d=1:m
                        theta(l,d)=(sum(W(:,l).*X(:,d)))/sum(W(:,l));
                    end
                end
                val1=theta;

                %% Weight and theta iterations in power k-means
                for iteration=1:60
                    md = pdist2(X,theta,'squaredeuclidean');
                    coef=(sum(md.^s,2)).^((1/s)-1);
                    for i=1:n
                        for l=1:p
                            W(i,l)=((md(i,l)+0.000001)^(s-1))*coef(i);
                        end
                    end

                    for l=1:p
                        for d=1:m
                            theta(l,d)=(sum(W(:,l).*X(:,d)))/sum(W(:,l));
                        end
                    end

                    if (sum(isnan(theta))~=0)
                        s_not_too_small=0;
                        disp('NA');
                        norm_flg=0;
                        [W,obj]=calculate_k_means_obj_func_value(X,n,m,p,W);
                        break
                    end

                    if (norm(val1-theta,'fro')<0.001)
                        break
                    else
                        val1=theta;
                    end
                end

                if norm_flg
                    [W,obj]=calculate_k_means_obj_func_value(X,n,m,p,W);
                end

                W2(:,(nn_count-1)*p+1:nn_count*p)=W;
                
                if obj<pbest(1,nn_count)
                    pbest(1,nn_count)=obj;
                    pbestx(:,(nn_count-1)*p+1:nn_count*p)=W;
                end
            end

            %% Determine the group best solution
            [~,pc]=min(pbest);
            if pbest(1,pc)<gbest
                gbest=pbest(1,pc);
                gbestx=repmat(pbestx(:,(pc-1)*p+1:pc*p),1,N);
                count_gli=0;
                fprintf('objective function is updated: %0.8f\n',gbest)
                %disp(gbest)
            else
                count_gli=count_gli+1;
            end

            %% Termination Criterion - CNO
            if count_gli>M
                break
            end

            %% Compute the diversity and perform mutation if necessary
            Div=norm(W2-gbestx,'fro');
            Div=Div/(N*n*p);
            it=it+1;
            if Div < 0.001
                disp('mutation!')
                a=exp(it/1000);%1-e
                psi=-2.5*a+5*a*rand(n,N*p);%-2.5a-2.5a
                et=(1/sqrt(a))*exp((-(psi/a).^2)/2).*cos((5*(psi/a)));
                larger_zero=et>0;
                lower_zero=et<0;
                W2=W2+larger_zero.*et.*(1-W2);
                W2=W2+lower_zero.*et.*W2;

                v=2*rand(n,N*p)-1;
                pbest=1e10*ones(1,N);
            else
                %% Reposition the initial states by using a PSO rule
                v=w*v+beta1*rand(n,N*p).*(pbestx-W2)+beta2*rand(n,N*p).*(gbestx-W2);
                W2=W2+v;
                W2=min(1,max(0,W2));
            end
        end
    else
        while true
            %% Multiple models for scattered local search and determining the individual best. 
            % The power kmeans is degenerated to a kmeans due to a small value of s
            for nn_count=1:N
                W=W2(:,(nn_count-1)*p+1:nn_count*p);
                %                 [~,label]=max(W,[],2);
                %                 for l = 1:p
                %                     theta(l, :) = mean(X(label == l, :),1);
                %                 end
                W=W';
                theta = W*X./(sum(W,2)*ones(1,size(X,2))); %new center

                index=isnan(sum(theta,2));
                if sum(index)~=0
                    theta(index,:)=X(randperm(n,sum(index)),:);
                end

                md = pdist2(X,theta,'squaredeuclidean');
                [md_min,label]=min(md,[],2);
                pre_obj=sum(md_min);

                while 1
                    for l = 1:p
                        theta(l, :) = mean(X(label == l, :),1);
                    end
                    md = pdist2(X,theta,'squaredeuclidean');
                    [md_min,label]=min(md,[],2);
                    obj=sum(md_min);

                    if pre_obj-obj<1e-10
                        break
                    else
                        pre_obj=obj;
                    end
                end
                W=zeros(n,p);
                for i=1:n
                    W(i,label(i))=1;
                end

                W2(:,(nn_count-1)*p+1:nn_count*p)=W;
                if obj<pbest(1,nn_count)
                    pbest(1,nn_count)=obj;
                    pbestx(:,(nn_count-1)*p+1:nn_count*p)=W;
                end
            end

            %% Determine the group best solution
            [~,pc]=min(pbest);
            if pbest(1,pc)<gbest
                gbest=pbest(1,pc);
                gbestx=repmat(pbestx(:,(pc-1)*p+1:pc*p),1,N);
                count_gli=0;
                fprintf('objective function is updated: %0.8f\n',gbest)
                %disp(gbest)
            else
                count_gli=count_gli+1;
            end

            %% Termination Criterion - CNO
            if count_gli>M
                break
            end

            %% Compute the diversity and perform mutation if necessary
            Div=norm(W2-gbestx,'fro');
            Div=Div/(N*n*p);
            it=it+1;
            if Div < 0.001
                disp('mutation!')
                a=exp(it/1000);%1-e
                psi=-2.5*a+5*a*rand(n,N*p);%-2.5a-2.5a
                et=(1/sqrt(a))*exp((-(psi/a).^2)/2).*cos((5*(psi/a)));
                larger_zero=et>0;
                lower_zero=et<0;
                W2=W2+larger_zero.*et.*(1-W2);
                W2=W2+lower_zero.*et.*W2;
                v=2*rand(n,N*p)-1;
                pbest=1e10*ones(1,N);
            else
                %% Reposition the initial states by using a PSO rule
                v=w*v+beta1*rand(n,N*p).*(pbestx-W2)+beta2*rand(n,N*p).*(gbestx-W2);
                W2=W2+v;
                W2=min(1,max(0,W2));
            end
        end
    end
    
    %% If the s is smaller than s_min, end the loop
    if (s_not_too_small==0)&&(s<-300)      
            fprintf('break in step %d\n',t)
            break
    end

    %% Annealing the power parameter s
    s=eta*s;
    fprintf('Reduce s: %f\n',s)
end
disp('end')

W=gbestx(:,1:p);
[~,label]=max(W,[],2);
end

function [W,obj]=calculate_k_means_obj_func_value(X,n,m,p,W)
[~,label]=max(W,[],2);
theta=zeros(p,m);
for l = 1:p
    theta(l, :) = mean(X(label == l, :),1);
end
index=isnan(sum(theta,2));
if sum(index)~=0
    theta(index,:)=X(randperm(n,sum(index)),:);
end

md = pdist2(X,theta,'squaredeuclidean');
[md_min,label]=min(md,[],2);
pre_obj=sum(md_min);

while 1
    for l = 1:p
        theta(l, :) = mean(X(label == l, :),1);
    end
    md = pdist2(X,theta,'squaredeuclidean');
    [md_min,label]=min(md,[],2);
    obj=sum(md_min);

    if pre_obj-obj<1e-10
        break
    else
        %disp(pre_obj-obj)
        pre_obj=obj;
    end
end
W=zeros(n,p);
for i=1:n
    W(i,label(i))=1;
end
end

function [W2,pbest,pbestx,gbest,gbestx]=generate_centers_and_assign_w_initial_pbest_gbest(X,n,p,N,pbest,pbestx,gbest,gbestx)
sz=[n,p];
row=[1:n];

for nn_count=1:N
    % initialization of centers using kmeans++
    % sam=randperm(n,p);
    % theta=X(sam,:);    
    theta=initialization_kpp(X, p);

    md = pdist2(X,theta,'squaredeuclidean');
    [~,label]=min(md,[],2);
    [obj,~,label] = label2wd_2(X,label,p);
    pre_obj=obj;
    while 1
        for l = 1:p
            theta(l, :) = mean(X(label == l, :),1);
        end
        md = pdist2(X,theta,'squaredeuclidean');
        [md_min,label]=min(md,[],2);
        obj=sum(md_min);
        if pre_obj-obj<1e-10
            break
        else
            pre_obj=obj;
        end
    end

    W=zeros(n,p);
    W(sub2ind(sz,row,label'))=1;
    pbest(1,nn_count)=obj;
    pbestx(:,(nn_count-1)*p+1:nn_count*p)=W;

end
W2=pbestx;

[best,pc]=min(pbest);
if best<gbest
    gbest=best;
    gbestx=repmat(pbestx(:,(pc-1)*p+1:pc*p),1,N);
    fprintf('objective function is updated in initialization: %0.8f\n',gbest)
end

for i=1:size(W2,2)
    if all((W2(:,i))==0)
        W2(randperm(n,p),i)=1;
    end
end
end

function centroid=initialization_kpp(data, k)
% Choose the first inital centroid randomly
centroid = data(randperm(size(data,1),1)',:);

% Select remaining initial centroids (a total number of k-1)
for i = 2:k
    distance_matrix = zeros(size(data,1),i-1);
    for j = 1:size(distance_matrix,1)
        for p = 1:size(distance_matrix,2)
            distance_matrix(j,p) = sum((data(j,:)-centroid(p,:)) .^ 2);
        end
    end
    % Choose next centroid according to distances between points and
    % previous cluster centroids.
    index = Roulettemethod(distance_matrix);
    centroid(i,:) = data(index,:);
end
end

function [index] = Roulettemethod(distance_matrix)

% Find shortest distance between one sample and its closest cluster centroid
[min_distance,~] = min(distance_matrix,[],2);

% Normalize for further operations
min_distance = min_distance ./ sum(min_distance);

% Construct roulette according to min_distance
temp_roulette = zeros(size(distance_matrix,1),1);
for i = 1:size(distance_matrix,1)
    temp_roulette(i,1) = sum(min_distance(1:i,:));
end

% Generate a random number for selection
temp_rand = rand();

% Find the corresponding index
for i = 1:size(temp_roulette,1)
    if((i == 1) && temp_roulette(i,1) > temp_rand)
        index = 1;
    elseif((temp_roulette(i,1) > temp_rand) && (temp_roulette(i-1,1) < temp_rand))
        index = i;
    end
end
end
