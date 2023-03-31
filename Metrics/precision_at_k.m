function [score] = precision_at_k(Ygt,Ypred,k)
%
% Average Precision@k (P@k) for multi-label classification
%
chk = sum(sum(Ygt,2)<k);
if(chk>0)
    warning(['#ground truth labels are less than k = ' num2str(k) ' for ',...
        num2str(chk) ' instances.'])
end

[N,~] = size(Ygt);
score = 0;
Nd = N;
for ii = 1:N
    if(sum(Ygt(ii,:))>0)
        [~,inds] = sort(Ypred(ii,:),'descend');
        inds_k = inds(1:k);
        score = score + sum(Ygt(ii,inds_k))/k;
    else
        Nd = Nd - 1;
    end
end
score = score/Nd;