function [cdist] = nemenyi_cd(k,N,ALPHA)
%
% Computes Nemenyi critical distance (see [1]). q_alpha_vals are from [1].
%
%[1] J. Demsar. Statistical Comparisons of Classifiers
% over Multiple Data Sets. JMLR. 2006
%
if(k>1 && k<11)
    if(ALPHA == 0.05)
         q_alpha_vals = [1.960 2.343 2.569 2.728 2.850 2.949 3.031 3.102 3.164];
         q_alpha = q_alpha_vals(k-1);
    elseif(ALPHA == 0.1)
         q_alpha_vals = [1.645 2.052 2.291 2.459 2.589 2.693 2.780 2.855 2.920];
         q_alpha = q_alpha_vals(k-1);
    else
        error(['ALPHA = ' num2str(ALPHA) ' paramater not valid!']) 
    end
    
    cdist = q_alpha*sqrt(k*(k+1)/(6*N));

else
    error(['Can not compute critical distance for k = ' num2str(k)])    
end