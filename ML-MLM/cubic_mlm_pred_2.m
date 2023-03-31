function [Yscore]  = cubic_mlm_pred_2(Dxtest,T,Beta)
%
% Cubic MLM prediction with precomputed distance matrix Dxtest
%
N = size(Dxtest,1); 
Deltahat = Dxtest*Beta;

Yscore = zeros(N,1);
k = length(Deltahat(1,:));
for ii=1:N
    Deltak = Deltahat(ii,:)'; %k x 1
    ypred = cubic_eq_solver(T,Deltak,k);
    Yscore(ii,:) = ypred;
end

end


function y_out = cubic_eq_solver(T,deltahat,K)
    %
    % 
    %
    deltahat2 = deltahat.^2;
    
    % Third order polynomial coefficients for the deritative of the cost
    % function (11)
    A = K; 
    B = -3*sum(T);
    C = sum(3*(T.^2)-deltahat2); 
    D = sum(deltahat2.*T-T.^3);
    p = [A B C D];
    
    % Find three roots
    R = roots(p);
    I = [isreal(R(1));isreal(R(2));isreal(R(3))];
    
    % Select a root corresponding to smallest cost function value
    Jy = ones(3,1)*realmax;
    if(R(1)~= R(2) && I(1) && I(2))
        for j = 1:3
            Jy(j) = sum(((R(j)-T).^2-deltahat2).^2);
        end
        [~,ind] = min(Jy);
        y_out = R(ind);
    else
        y_out = R(I);
        y_out = y_out(1);
    end
end