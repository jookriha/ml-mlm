function [score] = label_cardinality(Y)
%
% Computes label cardinality for label matrix Y
%
N = size(Y,1);
score = sum(sum(Y,2))/N;

