function [Ypred,Dy_pred] = nn_mlm_pred(X,B,R,T)
%
% Nearest Neighbour Minimal Learning Machine (MLM) output prediction for multi-label 
% classification.
%
Dx = pdist2(X,R);
Dy_pred = Dx*B;
[~,inds] = min(Dy_pred,[],2);
Ypred = T(inds,:);