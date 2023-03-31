function [score,score_arr] = average_precision(Ygt,Ypred)
%
% Average precision for multi-label classification
%

[N,~] = size(Ygt);
score = 0;
score_arr = zeros(N,1);
Nd = N;
for ii = 1:N
    Lii = sum(Ygt(ii,:));
    if(Lii>0)
        [~,inds] = sort(Ypred(ii,:),'descend');
        score_arr(ii) = sum((1:Lii)./find(Ygt(ii,inds)==1))/Lii;
        score = score + score_arr(ii);
    else
        Nd = Nd - 1;
    end
end
score = score/Nd;