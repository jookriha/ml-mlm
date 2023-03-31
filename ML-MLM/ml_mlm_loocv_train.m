function [p_best,tc,score] = ml_mlm_loocv_train(hat_matrix,Dy,Xtrain,Ytrain,p_grid)
%
% ML-MLM training with LOOCV using PRESS ranking loss statistic
%
Yhat = hat_matrix*Dy;
N = size(Xtrain,1);

% Out-of-sample distance estimates via LOOCV
Dy_pred_PRESS = abs((Yhat-diag(hat_matrix).*Dy)./(ones(N,1)-diag(hat_matrix)));

score = zeros(length(p_grid),1);
rloss_best = realmax;
for p_ind = 1:length(p_grid)
    p = p_grid(p_ind);
    Rho_inv = 1./(Dy_pred_PRESS.^p); % Rho_inv is K x N
    r_sum = sum(Rho_inv,2); % r_sum is N x 1
    maxrho = max(Rho_inv,[],2);% maxrho is N x 1

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
    W = Rho_inv./r_sum;
    Yscore = W*Ytrain;
    [score(p_ind),~] = ranking_loss(Ytrain,Yscore);
    fprintf('p = %f, rloss statistic =  %f\n', p, score(p_ind));

    if(score(p_ind)<rloss_best)
       rloss_best = score(p_ind);
       Yscore_temp = Yscore;
    end
end
    
% Select an optimized ML-MLM model according to the smallest LOOCV ranking loss statistic
[~, min_p_ind] = min(score);
p_best = p_grid(min_p_ind);

% Cardinality-based thresholding selection for the optimized ML-MLM model
CtrN = sum(Ytrain(:));
Ytc = sort(Yscore_temp(:),'descend');
tc = (Ytc(CtrN) + Ytc(CtrN+1))/2;