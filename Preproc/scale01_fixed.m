function [Xs] = scale01_fixed(X,Xmax,Xmin,Iconst)
%
% Scales data set X to the range of [0,1]
% with given Xmax Xmin Iconst parameters
% values differing from constant variables are set to 1
%
Xrange = Xmax-Xmin;
[N,M] = size(X);
Xs = zeros(N,M);
if(sum(Iconst)>0)
    warning('Scaling requires constant variable handling.') 
    Xs(:,~Iconst) = (X(:,~Iconst) - repmat(Xmin(~Iconst),[N,1]))./...
    repmat(Xrange(~Iconst),[N,1]);
    Xs(:,Iconst) = abs(X(:,Iconst) - repmat(Xmax(Iconst),[N,1])) > 0;
else
   Xs = (X - repmat(Xmin,[N,1]))./repmat(Xrange,[N,1]); 
end