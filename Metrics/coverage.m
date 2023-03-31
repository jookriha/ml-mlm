function [score,score_arr] = coverage(Ygt,Ypred)
%
% Coverage for multi-label classification
%

[N,~] = size(Ygt);
score = 0;
score_arr = zeros(N,1);
Nd = N;
for ii = 1:N
    if(sum(Ygt(ii,:))>0)
        [~,inds] = sort(Ypred(ii,:),'descend');
        coverage_ii = find(Ygt(ii,inds),1,'last') - 1;
        score_arr(ii) = coverage_ii;
        score = score + coverage_ii;
    else
        Nd = Nd - 1;
    end
end
score = score/Nd;