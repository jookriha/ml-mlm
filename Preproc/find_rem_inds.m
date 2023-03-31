function [Idp,I] = find_rem_inds(X)
%
% Finds a set of unique observations for matrix X and
% returns indices of removable observations Idp and unique ones I
%
N = size(X,1);
[~,I,~] = unique(X,'rows');
I = sort(I);
Idp = setdiff(1:N,I)';