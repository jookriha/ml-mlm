function [score] = label_density(Y)
%
% Computes label density for label matrix Y
%
[N,L] = size(Y);
score = sum(sum(Y,2)/L)/N;

