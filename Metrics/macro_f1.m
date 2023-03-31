function [score] = macro_f1(Ygt,Ypred)
%
% Macro F1 for multi-label classification
%

[~,L] = size(Ygt);
score = 0;
for ii = 1:L
    TP = Ygt(:,ii)'*Ypred(:,ii);
    FP = sum(Ypred(:,ii))-TP;
    FN = sum(Ygt(:,ii))-TP;
    
    if(TP == 0)
        score = score + 0;
    else     
        precision_ii = TP/(TP+FP);
        recall_ii = TP/(TP+FN);
        score = score + 2*precision_ii*recall_ii/(precision_ii + recall_ii);
    end    
end
score = score/L;