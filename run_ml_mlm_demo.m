clear, close all
%
% Sample script for ML-MLM training and testing
%
addpath('./Metrics')
addpath('./Preproc')
addpath('./ML-MLM')

% load test data
dataname = 'SYNTHETIC';
SCALING = true;

metrics_str = {'ACCURACY','HAMMING_LOSS','MICROF1','MACROF1','RANKING_LOSS',...
    'COVERAGE','ONE_ERROR','AVERAGE_PRECISION'};

disp(['Loading ' num2str(dataname) 'data set...'])
load(['./INPUT/' dataname '/MAT-FORMAT/' dataname '.mat'],...
    'Xtrain','Ytrain','Xtest','Ytest')

if(SCALING)
    disp('Minmax-scaling input data set...')
    [Xtrain,Xmax,Xmin,Iconst] = scale01(Xtrain);
    Xtest = scale01_fixed(Xtest,Xmax,Xmin,Iconst);
end

[N,M] = size(Xtrain);
[~,L] = size(Ytrain);

grid_temp = 3:0.05:6;
p_grid = 2.^grid_temp;

% Distance regression model training with Moore-Penrose pseudoinverse.
[Dx,Irem,Ik,R,K,dx_alpha,Dy,P,hat_matrix,B_p] = dist_reg_train(Xtrain,Ytrain);

% Compute Local Rcut thresholding values
[Klr] = local_rcut(Ytrain,pdist2(Xtest,R)*B_p);

%
% ML-MLM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Running LOOCV ML-MLM training...')
[p_best,tc] = ml_mlm_loocv_train(hat_matrix,Dy,Xtrain,Ytrain,p_grid);

[ml_mlm.Ypred,ml_mlm.Yscore] = ml_mlm_pred(Xtest,R,Ytrain,B_p,p_best,tc);
ml_mlm.pms = compute_metrics(Ytest,ml_mlm.Ypred,ml_mlm.Yscore,metrics_str);

disp(['Selected power parameter: p = ' num2str(p_best)])
disp(['Selected thresholding: tc = ' num2str(tc)])

%
% BR-MLM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ntest = size(Xtest,1);
br_mlm.Yscore = zeros(Ntest,L);
br_mlm.Ypred = zeros(Ntest,L);
Dxtest = pdist2(Xtest,R);
T = Ytrain;
for ll = 1:L
    disp(['CMLM: Training for variable ' num2str(ll) '...'])
    Dy_ll = pdist2(Ytrain(:,ll),Ytrain(:,ll));
    B1 = P*(Dx'*Dy_ll);
    disp(['CMLM: predicting for variable ' num2str(ll) '...'])
    [br_mlm.Yscore(:,ll)] = cubic_mlm_pred_2(Dxtest,T(:,ll),B1);
end

br_mlm.Ypred = zeros(Ntest,L);
for ii=1:Ntest
    [~,inds] =  sort(br_mlm.Yscore(ii,:),'descend');
    br_mlm.Ypred(ii,inds(1:Klr(ii))) = 1;
end

br_mlm.pms = compute_metrics(Ytest,br_mlm.Ypred,br_mlm.Yscore,metrics_str);

% NN-MLM
disp('Running NN-MLM...')
nn_mlm.Ypred = nn_mlm_pred(Xtest,B_p,R,T);
nn_mlm.pms = compute_metrics(Ytest,nn_mlm.Ypred,[],metrics_str);

% LLS-MLM
disp('Running LLS-MLM...')
[lls_mlm.Ypred,lls_mlm.Yscore] = lls_mlm_pred(Xtest,R,T,B_p);
lls_mlm.pms = compute_metrics(Ytest,lls_mlm.Ypred,lls_mlm.Yscore,metrics_str);

disp('-------------------------------------')
disp('Accuracy')
disp(['lls-mlm: ' num2str(lls_mlm.pms.acc)])
disp(['ml-mlm:  ' num2str(ml_mlm.pms.acc)])
disp(['br-mlm:  ' num2str(br_mlm.pms.acc)])
disp(['nn-mlm:  ' num2str(nn_mlm.pms.acc)])
disp('-------------------------------------')
disp('Hamming loss')
disp(['lls-mlm: ' num2str(lls_mlm.pms.hl)])
disp(['ml-mlm:  ' num2str(ml_mlm.pms.hl)])
disp(['br-mlm:  ' num2str(br_mlm.pms.hl)])
disp(['nn-mlm:  ' num2str(nn_mlm.pms.hl)])
disp('-------------------------------------')
disp('Micro f1')
disp(['lls-mlm: ' num2str(lls_mlm.pms.micf1)])
disp(['ml-mlm:  ' num2str(ml_mlm.pms.micf1)])
disp(['br-mlm:  ' num2str(br_mlm.pms.micf1)])
disp(['nn-mlm:  ' num2str(nn_mlm.pms.micf1)])
disp('-------------------------------------')
disp('Macro f1')
disp(['lls-mlm: ' num2str(lls_mlm.pms.macf1)])
disp(['ml-mlm:  ' num2str(ml_mlm.pms.macf1)])
disp(['br-mlm:  ' num2str(br_mlm.pms.macf1)])
disp(['nn-mlm:  ' num2str(nn_mlm.pms.macf1)])

disp('-------------------------------------')
disp('Ranking loss')
disp(['lls-mlm: ' num2str(lls_mlm.pms.rl)])
disp(['ml-mlm:  ' num2str(ml_mlm.pms.rl)])
disp(['br-mlm:  ' num2str(br_mlm.pms.rl)])
disp('-------------------------------------')
disp('Coverage')
disp(['lls-mlm: ' num2str(lls_mlm.pms.cov)])
disp(['ml-mlm:  ' num2str(ml_mlm.pms.cov)])
disp(['br-mlm:  ' num2str(br_mlm.pms.cov)])
disp('-------------------------------------')
disp('One error')
disp(['lls-mlm: ' num2str(lls_mlm.pms.oe)])
disp(['ml-mlm:  ' num2str(ml_mlm.pms.oe)])
disp(['br-mlm:  ' num2str(br_mlm.pms.oe)])
disp('-------------------------------------')
disp('Average precision')
disp(['lls-mlm: ' num2str(lls_mlm.pms.ap)])
disp(['ml-mlm:  ' num2str(ml_mlm.pms.ap)])
disp(['br-mlm:  ' num2str(br_mlm.pms.ap)])