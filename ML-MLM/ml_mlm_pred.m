function [Ypred,Yscore] = ml_mlm_pred(Xtest,R,T,B,p,tc)
%
% Multi-Label Minimal Learning Machine prediction
%
N = size(Xtest,1);
disp(['Computing prediction for a test set N = ' num2str(N)])

% Predict distances in label space
pred_dists = pdist2(Xtest,R)*B;
  
% Prepare inverse distance weighting components
Rho_inv = 1./(abs(pred_dists).^p); % Rho_inv is N x K
r_sum = sum(Rho_inv,2); % r_sum is K x 1
maxrho = max(Rho_inv,[],2);% maxrho is K x 1

% Inf handling for larger power parameter values (typically when  p ~ 100)       
if(max(maxrho)==inf)
    Im = find(maxrho==inf);
    warning(['Handling close to zero distances for ' num2str(length(Im)) ' cases'])
    for ii = 1:length(Im)
        Rho_inv(Im(ii),Rho_inv(Im(ii),:)==inf) = 1;
        Rho_inv(Im(ii),Rho_inv(Im(ii),:)~=inf) = 0;
        r_sum(Im(ii)) = sum(Rho_inv(Im(ii),:));
    end
end
Irs = find(r_sum==inf);
if(~isempty(Irs))
     warning(['Handling inf divisor for ' num2str(length(Irs)) ' cases'])
    r_sum(Irs) = maxrho(Irs);
end

% Scale the weights
W = Rho_inv./r_sum;

% Construct convex combinations of label vectors (label scoring)
Yscore = W*T;

% Global thresholding (classification)
Ypred = Yscore>tc;