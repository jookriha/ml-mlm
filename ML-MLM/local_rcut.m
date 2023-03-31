function [Klr] = local_rcut(T,pred_dists)
%
% Computes instance-wise Local Rcut values.
%
[~,min_inds] = min(pred_dists,[],2);
Ts = T(min_inds,:);
Klr = sum(Ts,2);