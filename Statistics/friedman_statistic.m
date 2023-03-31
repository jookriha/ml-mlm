function [FF,pval,degf] = friedman_statistic(xr)
%
% P values are based on corrected friedman's statistic [1].
% 
% [1] J. Demsar. Statistical Comparisons of Classifiers
% over Multiple Data Sets. JMLR. 2006
%
[N,k] = size(xr);
f1 = 12*N/(k*(k+1));
Rj = mean(xr,1);
xsiF2 = f1*(sum(Rj.^2)-(k*(k+1)^2)/4);
FF = (N-1)*xsiF2/(N*(k-1)-xsiF2);

degf.v1 = k-1; degf.v2 = (k-1)*(N-1);
pval = fcdf(FF,degf.v1,degf.v2,'upper');