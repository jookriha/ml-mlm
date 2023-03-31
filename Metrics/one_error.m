function [score] = one_error(Ygt,Yscore)
%
% One Error for multi-label classification
%
score = 1 - precision_at_k(Ygt,Yscore,1);