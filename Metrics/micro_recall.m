function [score] = micro_recall(Ygt,Ypred)
%
% Micro recall for multi-label classification
%

[~,L] = size(Ygt);
TP = zeros(1,L);
FN = zeros(1,L);
for ii = 1:L
    TP(ii) = Ygt(:,ii)'*Ypred(:,ii);
    FN(ii) = sum(Ygt(:,ii))-TP(ii);
end
score = sum(TP)/(sum(TP)+sum(FN));