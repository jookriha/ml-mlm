function [mlcpm] = compute_metrics(Ygt,Ypred,Yscore,metrics_str)
%
% Computes MLC metrics for given set of label scores
%
mlcpm.acc = [];
mlcpm.hl = [];
mlcpm.micf1 = [];
mlcpm.macf1 = [];
mlcpm.rl = [];
mlcpm.cov = [];
mlcpm.oe = [];
mlcpm.ap = [];
% Bipartition based metrics
if(~isempty(Ypred))
    if(any(strcmp(metrics_str,'ACCURACY')))
        mlcpm.acc = accuracy(Ygt,Ypred);
    end
    if(any(strcmp(metrics_str,'HAMMING_LOSS')))
         mlcpm.hl = hamming_loss(Ygt,Ypred);
    end
    if(any(strcmp(metrics_str,'MICROF1')))
         mlcpm.micf1 = micro_f1(Ygt,Ypred);
    end
    if(any(strcmp(metrics_str,'MACROF1')))
         mlcpm.macf1 = macro_f1(Ygt,Ypred);
    end
end

% Ranking based metrics
if(~isempty(Yscore))
    if(any(strcmp(metrics_str,'RANKING_LOSS')))
        mlcpm.rl = ranking_loss(Ygt,Yscore);
    end
    if(any(strcmp(metrics_str,'COVERAGE')))
         mlcpm.cov = coverage(Ygt,Yscore);
    end
    if(any(strcmp(metrics_str,'ONE_ERROR')))
         mlcpm.oe = 1 - precision_at_k(Ygt,Yscore,1);
    end
    if(any(strcmp(metrics_str,'AVERAGE_PRECISION')))
         mlcpm.ap = average_precision(Ygt,Yscore);
    end
end