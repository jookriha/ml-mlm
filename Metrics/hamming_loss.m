function [score] = hamming_loss(Ygt,Ypred)
%
% Hamming loss for multi-label classification
%

[N,~] = size(Ygt);
score = 0;
for ii = 1:N
    score = score + mean(abs(Ygt(ii,:)-Ypred(ii,:)));
end
score = score/N;