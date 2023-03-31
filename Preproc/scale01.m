function [Xs,Xmax,Xmin,Iconst] = scale01(X)
%
% Scales data set X to the range of [0,1]
% Constant variables are set to 0
[N,M] = size(X);
Xmin = min(X,[],1);
Xmax = max(X,[],1);
Xrange = Xmax-Xmin;
Iconst = ~(Xrange > 0);
Xs = zeros(N,M);
if(sum(Iconst)>0)
    warning('Data set contains constant variables.')
    warning(['#constant features/#features: ' num2str(sum(Iconst)) '/' ...
        num2str(length(Xmin))])
end
Xs(:,~Iconst) = (X(:,~Iconst) - repmat(Xmin(~Iconst),[N,1]))./...
    repmat(Xrange(~Iconst),[N,1]);