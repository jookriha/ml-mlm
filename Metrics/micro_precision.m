function [score,TP,FP] = micro_precision(Ygt,Ypred)
%
% Micro precision for multi-label classification
%

[~,L] = size(Ygt);
TP = zeros(1,L);
FP = zeros(1,L);
for ii = 1:L
    TP(ii) = Ygt(:,ii)'*Ypred(:,ii);
    FP(ii) = sum(Ypred(:,ii))-TP(ii);
end
score = sum(TP)/(sum(TP)+sum(FP));