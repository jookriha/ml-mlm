function [score,score_arr] = ranking_loss(Ygt,Ypred)
%
% Ranking Loss for multi-label classification
%

[N,M] = size(Ygt);
score = 0;
score_arr = zeros(N,1);
Nd = N;
for ii = 1:N
    Lii = sum(Ygt(ii,:));
    if(Lii>0)
        [~,inds] = sort(Ypred(ii,:),'descend');
        ytemp = Ygt(ii,inds); ytemp_search = ~ytemp;
        rankloss_temp = 0;
        while(sum(ytemp_search) > 0)
            ind_temp = find(ytemp_search,1,'first');
            rankloss_temp = rankloss_temp+ sum(ytemp(ind_temp:end));
            ytemp(1:ind_temp) = [];
            ytemp_search = ~ytemp;
        end
        rankloss_temp = rankloss_temp/(Lii*(M-Lii));
        score_arr(ii) = rankloss_temp;
        score = score + rankloss_temp;
    else
        Nd = Nd - 1;
    end
end
score = score/Nd;