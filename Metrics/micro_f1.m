function [score] = micro_f1(Ygt,Ypred)
%
% Micro F1 for multi-label classification
%
mrec_score = micro_recall(Ygt,Ypred);
mpre_score = micro_precision(Ygt,Ypred);
score = 2*mrec_score*mpre_score/(mrec_score + mpre_score);