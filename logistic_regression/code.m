%% Title: Logistic Regression classifier 
% Project: Assignment1 for pattern recognition
% Description: Document classification according to the appearance 
% frequencies of different words
% Author: Wang Jie
% Reference: http://zjjconan.github.io/articles/2015/04/Softmax-Regression-Matlab/
% http://deeplearning.stanford.edu/wiki/index.php/Exercise:Softmax_Regression
% Creation date: 2016-03-03
% Last modified: 2016-03-07

% load data
% .data: docId, wordId, wordCount; .label:docClassLabel(rowN=docId); map:
% classNum-className
% Specificly, train data: 11269 53975*1 documents,
clear;
train=load('data\train.data');
trainlabel=importdata('data\train.label');
%记录时间
tic
% calculate features
% feature= frequency of each word in each set
% the calculation deserves consideration
K=20;
m=numel(trainlabel);              %m: number of trainset doc
f=numel(unique(train(:,2)));  %f: total number of different words

% trainf ---->sparse matrix
C=[1:m,train(:,1)'];
R=[ones(1,m),train(:,2)'+1];
V=[ones(1,m),train(:,3)'];
trainf=sparse(R,C,V,f+1,m);

% initialization
stepsize=0.05;
endstep=5000;
lamda = 0.01;
%the initial random parameters are important for the effect!!!
theta=0.005*randn(K, f+1);
J_old = 0;
Jvalue=zeros(endstep,1);
traint=zeros(endstep,1);

for iter = 1:endstep
    %将样本的label变成one-hot类型的，one-hot是指每个样本的label长度都是K, 对于第i类样本，其label的第i
    %个元素为1，其他元素为0. 这样用groundTruth就可以通过矩阵运算快速计算出J和g
    groundTruth = full(sparse(trainlabel, 1:m, 1));  
    eta=bsxfun(@minus,theta*trainf,max(theta*trainf,[],1));  
    eta=exp(eta);  
    pij=bsxfun(@rdivide,eta,sum(eta));  
    J=-1./m*sum(sum(groundTruth.*log(pij)))+lamda/2*sum(sum(theta.^2));
    Jvalue(iter)=J;
    g=-1/m.*(groundTruth-pij)*trainf'+lamda.*theta;

    theta=theta-stepsize*g;
    
    [~,trainc]=max(theta*trainf,[],1);
    traint(iter)=sum(trainc'==trainlabel)/m;
    traint(iter)
    
    if abs(J - J_old)<10e-10 || traint(iter)>0.95 
        break;
    else
        J_old = J
    end
    
end

% test
% given the softmax hypothesis function, we can only compare the numerators
% for train data
[~,trainclass]=max(theta*trainf,[],1);
traintrue=sum(trainclass'==trainlabel)/m;  
trainerror=1-traintrue;

% for test data
test=load('data\test.data');
testlabel=importdata('data\test.label');
mtest=numel(testlabel); 

%从test中删除没在train里面的词
[a,~] = find(test(:,2) > f);
test(a,:) = [];

C_test=[1:mtest,test(:,1)'];
R_test=[ones(1,mtest),test(:,2)'+1];
V_test=[ones(1,mtest),test(:,3)'];
testf=sparse(R_test,C_test,V_test,f+1,mtest);

[~,testclass]=max(theta*testf,[],1);
testtrue=sum(testclass'==testlabel)/mtest;  
testerror=1-testtrue;

rate=[traintrue*100,testtrue*100,trainerror*100,testerror*100];
str=['The train accuracy of LogitRegClassifier is %2.1f%%\n',...
    'The test accuracy of LogitRegClassifier is %2.1f%%\n',...
    'The train error of LogitRegClassifier is %2.1f%%\n',...
    'The test error of LogitRegClassifier is %2.1f%%\n'];
fprintf(str,rate)   

toc
save('LogitRegClassifier.mat','theta')
