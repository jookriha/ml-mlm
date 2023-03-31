function [score] = accuracy(Ygt,Ypred)
%
% Accuracy for multi-label classification
%

[N,~] = size(Ygt);
score = 0;
for ii = 1:N
    if(sum(Ygt(ii,:)+Ypred(ii,:))==0)
        score = score + 1;
    else
    score = score + sum(Ygt(ii,:).*Ypred(ii,:))/...
        sum((Ygt(ii,:)+Ypred(ii,:)) > 0);
    end
end
score = score/N;