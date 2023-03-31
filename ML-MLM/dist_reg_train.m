function [Dx,Irem,Ik,R,K,dx_alpha,Dy,P,hat_matrix,B_p] = dist_reg_train(Xtrain,Ytrain)
%
% Distance regression model training with Moore-Penrose pseudoinverse
%
disp('Computing input space distance matrix...')
Dx = pdist2(Xtrain,Xtrain);
[Irem,Ik] = find_rem_inds(Xtrain);
Dx(:,Irem) = [];
R = Xtrain;
R(Irem,:) = [];
K = size(R,1);
dx_alpha = quantile(pdist(R),0.001);


disp('Computing output space distance matrix...')
Dy = pdist2(Ytrain,Ytrain);
disp('Computing pinv...')
P = pinv(Dx'*Dx + dx_alpha*eye(K));
disp('Computing hat matrix...')
hat_matrix = Dx*P*Dx'; % N x N
disp('Solving distance regression model coefficients...')
B_p = P*(Dx'*Dy);