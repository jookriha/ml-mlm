function [Ypred,Yscore] = lls_mlm_pred(Xtest,R,T,B)
%
% LLS-MLM prediction with Local Rcut thresholding for Multi-Label
% Classification.
%

N = size(Xtest,1);
L = size(T,2);

disp('Predicting output space distances...')
pred_dists = pdist2(Xtest,R)*B;
[~,min_inds] = min(pred_dists,[],2);
nntreshold = sum(T(min_inds,:),2);

Yscore = zeros(N,L);
Ypred = zeros(N,L);

disp('Computing predictions with LLS...')
for ii=1:N
    % Solve multilateration with LLS
    Yscore(ii,:) = solve_lls(T,pred_dists(ii,:)',min_inds(ii));
    % Score labels according to multilateration solution
    [~,inds] =  sort(Yscore(ii,:),'descend');
    % Threshold with Local Rcut
    Ypred(ii,inds(1:nntreshold(ii))) = 1;
end

end


function [yscore] = solve_lls(T,rho,istar)
%
% Solves multilateration with Localization Linear Systems (LLS) [1] approach. Note that 
% benchmark-anchor-node (BAN) is defined by istar index.
%
% [1] 
%

if(size(T,1) ~= size(rho,1))
    error(['Dimension mismatch for T and rho: T is ' num2str(size(T,1)) ' x ' ...
    num2str(size(T,1)) ' and rho is ' num2str(size(rho,1)) ' x ' num2str(size(rho,2))])
end

% BAN related distance estimate (rho_s) and BAN output space reference
% point t_s.
rho_s = rho(istar);
t_s = T(istar,:);

rho(istar) = [];
T(istar,:) = []; 
K = size(T,1);

% Construct coefficient matrix A and b
A = T-repmat(t_s,[K,1]);
b = 0.5.*(repmat(rho_s^2,[K,1]) + pdist2(T,t_s).^2 - rho.^2);

% Solve transition vector theta from A theta = b => theta = A^(-1) b with OLS
theta = A\b;

% Compute prediction via transition from BAN with theta
yscore = t_s + theta';

end